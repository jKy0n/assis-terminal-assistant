# ~/.assis/embed_etc.py
import os
import pathlib
import gzip
import hashlib
import chromadb
import concurrent.futures
from sentence_transformers import SentenceTransformer

MAX_THREADS = 8
MAX_SIZE = 1024 * 100
model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.HttpClient(host="localhost", port=8000)
collection = chromadb_client = chromadb.HttpClient(host="localhost", port=8000).get_or_create_collection(name="assis-docs")

CONFIG_FILES = [
    '/etc/fstab',
    '/etc/hostname',
    '/etc/pacman.conf',
    '/etc/resolv.conf',
    # '/etc/sudoers',
    '/etc/systemd/system.conf',
]

def is_valid_file(path):
    try:
        return os.path.isfile(path) and os.path.getsize(path) <= MAX_SIZE
    except:
        return False

def hash_file(path):
    try:
        with open(path, 'rb') as f: return hashlib.sha256(f.read()).hexdigest()
    except: return ''

def load_chunks(text, max_lines=20):
    lines = text.splitlines()
    return ['\n'.join(lines[i:i+max_lines]) for i in range(0, len(lines), max_lines)]

def index_etc_file(path):
    if not is_valid_file(path): return
    try:
        with open(path, 'r', errors='ignore') as f: text = f.read()
        file_hash = hash_file(path)
        chunks = load_chunks(text)
        embeddings = model.encode(chunks)
        ids = [f"etc_{path.replace('/', '_')}_{i}" for i in range(len(chunks))]
        metadatas = [{
            'source': path,
            'chunk_index': i,
            'file_hash': file_hash,
            'type': 'config'
        } for i in range(len(chunks))]
        collection.upsert(documents=chunks, ids=ids, embeddings=embeddings.tolist(), metadatas=metadatas)
        print(f"📄 [ETC] {path} ({len(chunks)} blocos)")
    except Exception as e:
        print(f"⚠️ Erro em etc {path}: {e}")

print("🔄 Indexando /etc ...")
with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
    executor.map(index_etc_file, CONFIG_FILES)
print("✅ /etc concluído")