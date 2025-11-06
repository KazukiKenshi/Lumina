import os
import json
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
            raw_docs = [
                d
                for d in raw_docs
                if any(c.lower() in d.metadata.get("source", "").lower() for c in hint_conditions)
            ] or raw_docs  # fallback if empty

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
