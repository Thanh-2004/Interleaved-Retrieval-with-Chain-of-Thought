from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
import uuid
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 1. Đọc và tách file corpus
def load_corpus(txt_path):
    with open(txt_path, encoding='utf-8') as f:
        raw = f.read()
    # Mỗi mục có dạng: "Title:<title>\nPassage:...<endofpassage>"
    entries = raw.split('Title:')
    docs = []
    for e in entries:
        if not e.strip(): continue
        title, rest = e.split('Passage:', 1)
        passages = rest.split('<endofpassage>')
        for p in passages:
            text = p.strip()
            if text:
                docs.append({
                    'title': title.strip(),
                    'passage': text.replace('\n', ' ').strip()
                })
    return docs


def main():
    # 2. Chunking 
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200,
        separators=["\n\n", "\n", ".", "?", "!", " "]
    )

    # 3. Init mô hình embedding
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # 4. Kết nối MongoDB
    client = MongoClient('mongodb://localhost:27017')
    # client = MongoClient('mongodb+srv://<username>:<password>@cluster0.xxxxxx.mongodb.net/rag_db?retryWrites=true&w=majority')

    db = client['rag_db']
    collection = db['documents']
    collection.drop()   

    # 5. Xử lý và lưu
    for doc in load_corpus('VDT2025_Multihop RAG dataset/multihoprag_corpus.txt'):
        for chunk in splitter.split_text(doc['passage']):
            embedding = model.encode(chunk).tolist()
            record = {
                '_id': str(uuid.uuid4()),
                'title': doc['title'],
                'chunk': chunk,
                'embedding': embedding
            }
            collection.insert_one(record)

if __name__ == "__main__":
    main()
