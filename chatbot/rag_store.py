import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json

class RAGDataStore:
    def __init__(self, base_dir="data", store_dir="vectorstores"):
        self.base_dir = base_dir
        self.store_dir = store_dir
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        os.makedirs(store_dir, exist_ok=True)

    def _load_texts(self, folder):
        """Load all text files from a given folder"""
        docs = []
        for fname in os.listdir(folder):
            path = os.path.join(folder, fname)
            if path.endswith(".txt"):
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()
                    docs.append(Document(page_content=text, metadata={"source": fname}))
        return docs

    def build_store(self, category):
        """Create and persist FAISS store for one category"""
        folder = os.path.join(self.base_dir, category)
        docs = self._load_texts(folder)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        split_docs = text_splitter.split_documents(docs)

        store = FAISS.from_documents(split_docs, self.embeddings)
        store.save_local(os.path.join(self.store_dir, category))
        print(f"✅ Saved vector store for {category} with {len(split_docs)} chunks")

    def load_store(self, category):
        """Load existing FAISS store"""
        path = os.path.join(self.store_dir, category)
        return FAISS.load_local(path, self.embeddings, allow_dangerous_deserialization=True)

    def retrieve(self, query, category="main", k=3):
        """Perform retrieval"""
        store = self.load_store(category)
        docs = store.similarity_search(query, k=k)
        return "\n\n---\n\n".join([doc.page_content for doc in docs])
