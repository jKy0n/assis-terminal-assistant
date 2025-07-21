# ~/.assis/embed_home.py
import os
import pathlib
import hashlib
import chromadb
import concurrent.futures
from sentence_transformers import SentenceTransformer

# Initial Setup
MAX_THREADS = 8
MAX_SIZE = 1024 * 100  # 100 KB
BATCH_SIZE = 256

# Exclude directories/files
EXCLUDE_DIRS = [
    ".cache",
    ".cargo",
    ".config/obsidian/Cache",
    ".mozilla",
    ".npm",
    ".steam",
    "BraveSoftware",
    "chromium",
    "Code",
    "Code - OSS",
    "GitHub Desktop",
    "obsidian",
    "Rambox",''
    "rambox",
    "teams-for-linux",
]

model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.HttpClient(host="localhost", port=8000)
collection = client.get_or_create_collection(
  name="assis-docs",
  metadata={
    "hnsw:num_threads": 8,
    "hnsw:construction_ef": 128,
    "hnsw:search_ef": 64
  }
)

def is_valid_file(path):
    try:
        return (
            os.path.isfile(path)
            and os.path.getsize(path) <= MAX_SIZE
            and not os.path.basename(path).startswith(".")
        )
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

def should_exclude(path):
    return any(part in EXCLUDE_DIRS for part in pathlib.Path(path).parts)

def process_file(path):
    if not is_valid_file(path) or should_exclude(path):
        return []
    try:
        with open(path, 'r', errors='ignore') as f:
            text = f.read()
        file_hash = hash_file(path)
        chunks = load_chunks(text)
        return [(f"home_{path.replace('/', '_')}_{i}", chunk, {
            'source': path,
            'chunk_index': i,
            'file_hash': file_hash,
            'type': 'home'
        }) for i, chunk in enumerate(chunks)]
    except Exception as e:
        print(f"⚠️ Erro em home {path}: {e}")
        return []

print("🔄 Indexando /home ...")

# Include directories and files
INCLUDE_DIRS = [
    ".config",
    ".local/share",
    ".bashrc",
    ".zshrc",
    ".profile",
    ".xinitrc",
    ".Xresources"
]

home_paths = []
for name in INCLUDE_DIRS:
    path = pathlib.Path.home() / name
    if path.is_file() and is_valid_file(path):
        home_paths.append(str(path))
    elif path.is_dir():
        for p in path.rglob("*"):
            if is_valid_file(p) and not should_exclude(p):
                home_paths.append(str(p))

# Coleta documentos com paralelismo
documents = []
with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
    for result in executor.map(process_file, home_paths):
        if result:
            documents.extend(result)

# Indexação por lote com embeddings
for i in range(0, len(documents), BATCH_SIZE):
    batch = documents[i:i+BATCH_SIZE]
    ids = [x[0] for x in batch]
    texts = [x[1] for x in batch]
    metas = [x[2] for x in batch]
    try:
        embeddings = model.encode(texts, show_progress_bar=False, batch_size=32)
        collection.upsert(
            documents=texts,
            ids=ids,
            embeddings=embeddings,
            metadatas=metas
        )
        print(f"📄 [HOME] Lote {i // BATCH_SIZE + 1} de {len(documents) // BATCH_SIZE + 1} indexado")
    except Exception as e:
        print(f"❌ Erro no lote {i}: {e}")

print("✅ /home concluído")