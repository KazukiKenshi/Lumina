from langchain_community.embeddings import HuggingFaceEmbeddings
import numpy as np
class SymptomMatcher:
    def __init__(self):
        self.embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.symptom_db = {
            "depression": ["sad", "loss of interest", "hopeless", "fatigue", "worthless", "insomnia"],
            "anxiety": ["worry", "panic", "nervous", "heart racing", "can't relax", "fear"],
            "ocd": ["obsession", "compulsion", "checking", "cleaning", "intrusive thoughts"],
            "ptsd": ["flashbacks", "nightmares", "avoidance", "hypervigilant", "trauma"],
            "bipolar": ["manic", "elevated mood", "less sleep", "racing thoughts", "impulsive"],
        }
        self.embeddings = {k: self.embedder.embed_documents(v) for k, v in self.symptom_db.items()}
    def match(self, user_input: str):
        query_emb = np.array(self.embedder.embed_query(user_input))
        scores = {}
        for condition, embs in self.embeddings.items():
            sim = np.mean([np.dot(query_emb, np.array(e)) / (np.linalg.norm(e)*np.linalg.norm(query_emb)) for e in embs])
            scores[condition] = sim
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)
