import os
import re
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder
from mistralai import Mistral


class RAGDataStore:
    def __init__(self, base_dir="data", store_dir="vectorstores"):
        self.base_dir = base_dir
        self.store_dir = store_dir
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        # Attempt to import rapidfuzz for fuzzy matching; fallback to simple edit distance
        try:
            from rapidfuzz import fuzz
            self.fuzz = fuzz
        except Exception:
            self.fuzz = None
        # Load tag index if present
        self.tags = {}
        tags_path = os.path.join(self.base_dir, 'tags.json')
        if os.path.exists(tags_path):
            try:
                with open(tags_path, 'r', encoding='utf-8') as f:
                    import json
                    self.tags = json.load(f)
                print(f"[RAG] Loaded tags for categories: {list(self.tags.keys())}")
            except Exception as e:
                print(f"[RAG][WARN] Failed to load tags.json: {e}")

        # Optional cross-encoder for reranking (fast and accurate)
        try:
            self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        except Exception as e:
            print("[WARN] Cross-encoder not loaded, falling back to FAISS ranking:", e)
            self.reranker = None

        # Optional summarization model (tiny and fast)
        try:
            mistral_key = os.getenv("MISTRAL_API_KEY")
            self.client = Mistral(api_key=mistral_key) if mistral_key else None
        except Exception:
            self.client = None

        os.makedirs(store_dir, exist_ok=True)

    # ------------------------
    # Data Loading and Building
    # ------------------------
    def _load_texts(self, folder):
        docs = []
        for fname in os.listdir(folder):
            path = os.path.join(folder, fname)
            if path.endswith(".txt"):
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()
                    docs.append(Document(page_content=text, metadata={"source": fname}))
        return docs

    def build_store(self, category):
        folder = os.path.join(self.base_dir, category)
        docs = self._load_texts(folder)
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        split_docs = splitter.split_documents(docs)

        store = FAISS.from_documents(split_docs, self.embeddings)
        store.save_local(os.path.join(self.store_dir, category))
        print(f"✅ Saved vector store for {category} with {len(split_docs)} chunks")

    def load_store(self, category):
        path = os.path.join(self.store_dir, category)
        return FAISS.load_local(path, self.embeddings, allow_dangerous_deserialization=True)

    # ------------------------
    # Context Summarization
    # ------------------------
    def _summarize(self, text, topic_hint=None):
        """Use Mistral to shorten and clean retrieved context."""
        if not self.client:
            return text[:2000]

        messages = [
            {
                "role": "system",
                "content": (
                    "Summarize the following text into concise, bullet-style diagnostic criteria "
                    "or coping guidelines. Remove redundancy and focus only on relevant facts."
                ),
            },
            {"role": "user", "content": f"Topic: {topic_hint or 'general'}\n{text[:4000]}"},
        ]
        try:
            resp = self.client.chat.complete(model="mistral-tiny", messages=messages)
            summary = resp.choices[0].message.content.strip()
            return summary[:2000]
        except Exception as e:
            print(f"[WARN] Summarization failed: {e}")
            return text[:2000]

    # ------------------------
    # Improved Retrieval
    # ------------------------
    def retrieve(self, query, category="diagnosis", k=5, hint_conditions=None, summarize=True):
        """Retrieve optimized context based on user symptoms and condition hints."""

        store = self.load_store(category)
        raw_docs = store.similarity_search(query, k=10)  # get a wide net

        # --- 1️⃣ Filter by hint conditions (if provided) ---
        if hint_conditions:
            # Expand filtering logic: match either filename or associated tags
            cat_tags = self.tags.get(category, {})
            def doc_matches(d):
                source = d.metadata.get('source', '').lower()
                # direct filename substring match
                if any(c.lower() in source for c in hint_conditions):
                    return True
                # tag match
                file_tags = [t.lower() for t in cat_tags.get(source, [])]
                return any(t in hint_conditions_lower for t in file_tags)

            hint_conditions_lower = [c.lower() for c in hint_conditions]
            filtered = [d for d in raw_docs if doc_matches(d)]
            if filtered:
                raw_docs = filtered

        # --- 2️⃣ Optional cross-encoder re-ranking ---
        if self.reranker:
            pairs = [[query, d.page_content] for d in raw_docs]
            scores = self.reranker.predict(pairs)
            ranked = [d for _, d in sorted(zip(scores, raw_docs), reverse=True)]
        else:
            ranked = raw_docs

        top_docs = ranked[:k]
        combined_text = "\n\n---\n\n".join([d.page_content[:800] for d in top_docs])
        combined_text = combined_text[:4000]  # hard cap

        # --- 3️⃣ Optional summarization ---
        if summarize:
            topic_hint = ", ".join(hint_conditions or [category])
            combined_text = self._summarize(combined_text, topic_hint)

        # --- 4️⃣ Return clean final text ---
        return combined_text.strip()

    # ------------------------
    # Tag Matching Helper
    # ------------------------
    def match_tags(self, query: str, category: str, min_len: int = 3):
        """Return matched tags using:
        1. Direct substring
        2. Stem/lemma normalization (simple heuristic)
        3. Fuzzy partial ratio / edit distance fallback
        """
        query_lower = query.lower()
        tokens = [t for t in re.split(r"\W+", query_lower) if t]

        def simple_stem(w: str) -> str:
            # Very lightweight stemmer (avoid heavy deps): strip common suffixes
            for suf in ("ingly", "edly", "ingly", "ing", "edly", "ed", "ness", "less", "ful", "ment", "tion", "sion", "ions", "tional", "s", "es"):
                if w.endswith(suf) and len(w) - len(suf) >= 3:
                    return w[: -len(suf)]
            return w

        token_stems = {simple_stem(t) for t in tokens}

        matched = set()
        for fname, tag_list in self.tags.get(category, {}).items():
            for tag in tag_list:
                tl = tag.lower()
                if len(tl) < min_len:
                    continue
                # Direct substring / phrase match
                if tl in query_lower:
                    matched.add(tl)
                    continue
                # Token / stem overlap (for multi-word tags split them)
                parts = [p for p in re.split(r"\W+", tl) if p]
                part_stems = {simple_stem(p) for p in parts}
                if part_stems & token_stems:
                    matched.add(tl)
                    continue
                # Fuzzy: compare each token to tag (or last word of tag) if rapidfuzz available
                if self.fuzz:
                    last = parts[-1] if parts else tl
                    for tok in tokens:
                        if len(tok) >= 3:
                            try:
                                score = self.fuzz.partial_ratio(tok, tl)
                            except Exception:
                                score = 0
                            if score >= 85:  # high-confidence fuzzy match
                                matched.add(tl)
                                break
                else:
                    # Simple edit distance fallback (Levenshtein) for last word only
                    def edit_distance(a, b):
                        m, n = len(a), len(b)
                        if m == 0: return n
                        if n == 0: return m
                        dp = [[0]*(n+1) for _ in range(m+1)]
                        for i in range(m+1): dp[i][0] = i
                        for j in range(n+1): dp[0][j] = j
                        for i in range(1,m+1):
                            for j in range(1,n+1):
                                cost = 0 if a[i-1]==b[j-1] else 1
                                dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+cost)
                        return dp[m][n]
                    last = parts[-1] if parts else tl
                    for tok in tokens:
                        if len(tok) >= 4:
                            dist = edit_distance(tok, last)
                            if dist <= 2:  # allow small typo
                                matched.add(tl)
                                break
        return list(matched)
