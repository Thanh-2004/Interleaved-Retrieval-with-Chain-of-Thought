from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
import numpy as np

# --- Khởi tạo model và kết nối DB như trước ---
model = SentenceTransformer('all-MiniLM-L6-v2')
client = MongoClient('mongodb://localhost:27017')
db = client['rag_db']
collection = db['documents']

# --- 1. Phương pháp brute-force 在 Python ---
def retrieve_bruteforce(query: str, top_k: int = 5):
    # 1. Compute query embedding & norm
    q_vec = model.encode(query)
    q_norm = np.linalg.norm(q_vec)
    
    # 2. Fetch all docs
    cursor = collection.find({}, {'title':1, 'chunk':1, 'embedding':1})
    
    # 3. Tính score và gom nhóm theo title
    best_per_title = {}  # title -> (score, chunk)
    for doc in cursor:
        title = doc['title']
        chunk = doc['chunk']
        emb = np.array(doc['embedding'], dtype=np.float32)
        score = np.dot(q_vec, emb) / (q_norm * np.linalg.norm(emb))
        
        # nếu title chưa có hoặc score hiện tại cao hơn thì cập nhật
        if title not in best_per_title or score > best_per_title[title][0]:
            best_per_title[title] = (score, chunk)
    
    # 4. Chuyển sang list và sort theo score giảm dần
    results = [
        (score, title, chunk)
        for title, (score, chunk) in best_per_title.items()
    ]
    results.sort(key=lambda x: x[0], reverse=True)
    
    # 5. Chỉ lấy top_k titles
    return results[:top_k]


# --- 2. Phương pháp Vector Search (MongoDB Atlas) ---
def retrieve_vector_search(query: str, top_k: int = 5):
    # 2.1. Tạo embedding
    q_vec = model.encode(query).tolist()
    
    # 2.2. Chạy pipeline kNN trên Atlas
    pipeline = [
        {
            '$search': {
                'knnBeta': {
                    'vector': q_vec,
                    'path': 'embedding',
                    'k': top_k
                }
            }
        },
        {
            '$project': {
                'title': 1,
                'chunk': 1,
                'score': {'$meta': 'searchScore'}
            }
        }
    ]
    return list(collection.aggregate(pipeline))


# --- Ví dụ sử dụng ---
if __name__ == '__main__':
    q = "Who is the individual associated with the cryptocurrency industry facing a criminal trial on fraud and conspiracy charges, as reported by both The Verge and TechCrunch, and is accused by prosecutors of committing fraud for personal gain?"
    
    print("=== Brute-force results ===")
    for score, title, chunk in retrieve_bruteforce(q, top_k=10):
        print(f"[{score:.4f}] {title}\n→ {chunk}\n")
    
    # # Nếu bạn có Atlas Vector Search:
    # print("=== Atlas Vector Search results ===")
    # for doc in retrieve_vector_search(q, top_k=3):
    #     print(f"[{doc['score']:.4f}] {doc['title']}\n→ {doc['chunk']}\n")
