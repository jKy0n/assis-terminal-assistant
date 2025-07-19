import os
import hashlib
import chromadb
import concurrent.futures
from sentence_transformers import SentenceTransformer, util

# Performance setup
MAX_THREADS = 8
MAX_SIZE = 1024 * 100
BATCH_SIZE = 256

# Include file list
CONFIG_FILES = [
    '/etc/fstab',
    '/etc/hostname',
    '/etc/pacman.conf',
    '/etc/resolv.conf',
    # '/etc/sudoers',
    '/etc/systemd/system.conf'
]

def is_valid_file(path):
    try:
        return os.path.isfile(path) and os.path.getsize(path) <= MAX_SIZE
    except:
        return False

def hash_file(path):
    try:
        with open(path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()
    except:
        return ""

def load_chunks(text, max_lines=20):
    lines = text.splitlines()
    return ['\n'.join(lines[i:i+max_lines]) for i in range(0, len(lines), max_lines)]

def process_etc_file(path):
    docs = []
    if not is_valid_file(path):
        return docs
    try:
        with open(path, 'r', errors='ignore') as f:
            text = f.read()
        file_hash = hash_file(path)
        for i, chunk in enumerate(load_chunks(text)):
            docs.append((
                f"etc_{path.replace('/', '_')}_{i}",
                chunk,
                {
                    'source': path,
                    'chunk_index': i,
                    'file_hash': file_hash,
                    'type': 'config'
                }
            ))
    except Exception as e:
        print(f"⚠️ Erro em etc {path}: {e}")
    return docs

if __name__ == "__main__":
    print("🔄 Indexando /etc ...")
    documents = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_THREADS) as exc:
        for doc_list in exc.map(process_etc_file, CONFIG_FILES):
            documents.extend(doc_list)

    model = SentenceTransformer("all-MiniLM-L6-v2")

    client = chromadb.HttpClient(host="127.0.0.1", port=8000)
    collection = client.get_or_create_collection(
    name="assis-docs",
    metadata={
        "hnsw:num_threads": 6,
        "hnsw:construction_ef": 128,
        "hnsw:search_ef": 64
    }
    )

    for i in range(0, len(documents), BATCH_SIZE):
        batch = documents[i:i+BATCH_SIZE]
        ids = [x[0] for x in batch]
        texts = [str(x[1]) for x in batch]
        metas = [x[2] for x in batch]

        try:
            embeddings = model.encode(texts, show_progress_bar=False, batch_size=32)
            collection.upsert(documents=texts, ids=ids, embeddings=embeddings, metadatas=metas)
            print(f"📄 [ETC] Lote {i//BATCH_SIZE+1}/{(len(documents)-1)//BATCH_SIZE+1} indexado")
        except Exception as e:
            print(f"❌ Erro no lote {i}: {e}")

    print("✅ /etc concluído")
