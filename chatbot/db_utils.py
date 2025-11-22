from pymongo import MongoClient
import os

def get_recent_exchanges(n=4, user_id=None):
    print(f"[DB DEBUG] get_recent_exchanges called with n={n}, user_id={user_id}")
    mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
    db_name = os.getenv("MONGO_DB", "lumina")
    collection_name = os.getenv("MONGO_COLLECTION", "chat_history")
    try:
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=3000)
        db = client[db_name]
        coll = db[collection_name]
        client.server_info()
    except Exception as conn_err:
        print(f"[DB ERROR] Could not connect to MongoDB: {conn_err}")
        return []
    query = {}
    if user_id:
        query["userId"] = user_id
    doc = coll.find_one(query, sort=[("updatedAt", -1)])
    exchanges = []
    if doc and "messages" in doc:
        msgs = doc["messages"]
        for msg in msgs[-n:]:
            exchanges.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", ""),
                "timestamp": str(msg.get("timestamp", ""))
            })
        print(f"[DB INFO] Loaded {len(exchanges)} recent exchanges for user {user_id or '[any]'} from latest chat history.")
    else:
        print(f"[DB INFO] No recent exchanges found for user {user_id or '[any]'} in chat_history collection.")
    return exchanges
