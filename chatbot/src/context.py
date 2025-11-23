import os
import sys
import io
import logging
from datetime import datetime
from dotenv import load_dotenv
from mistralai import Mistral

from src.rag_store import RAGDataStore
from src.symptom_reasoning import SymptomMatcher

# -------------------------
# Logger + terminal capture setup
# -------------------------
LOG_DIR = os.path.dirname(__file__)
LOG_FOLDER = os.path.join(LOG_DIR, 'logs')
os.makedirs(LOG_FOLDER, exist_ok=True)

logger = logging.getLogger('lumina_chatbot')
logger.setLevel(logging.INFO)
if not logger.handlers:
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s', datefmt='%H:%M:%S'))
    logger.addHandler(sh)

TERMINAL_CAPTURE_PATH = os.path.join(LOG_FOLDER, f"terminal-{datetime.now().strftime('%Y-%m-%d')}.txt")

class TeeStream(io.TextIOBase):
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            try:
                s.write(data)
            except Exception:
                pass
        return len(data)
    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass

try:
    terminal_file_handle = open(TERMINAL_CAPTURE_PATH, 'a', encoding='utf-8')
    sys.stdout = TeeStream(sys.stdout, terminal_file_handle)
    sys.stderr = TeeStream(sys.stderr, terminal_file_handle)
except Exception as e:
    logger.warning(f"Failed to set up terminal capture: {e}")

# -------------------------
# Core services: RAG system, symptom matcher, LLM client
# -------------------------
load_dotenv()

rag_system = RAGDataStore(base_dir="data", store_dir="vectorstores")
for domain in ["diagnosis", "counselling", "wellness"]:
    try:
        rag_system.build_store(domain)
    except Exception as e:
        logger.warning(f"Vector store build failed for {domain}: {e}")

matcher = SymptomMatcher()

client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

__all__ = ["logger", "rag_system", "matcher", "client"]
