# ~/.assis/embed_man.py
import pathlib
import gzip
import hashlib
import chromadb
import concurrent.futures
from sentence_transformers import SentenceTransformer

MAX_THREADS = 8
MAX_SIZE = 1024 * 500  # manpages podem ser maiores
model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.HttpClient(host="localhost", port=8000)
collection = chromadb_client = chromadb.HttpClient(host="localhost", port=8000).get_or_create_collection(name="assis-docs")

man_root = pathlib.Path('/usr/share/man')

print("🔄 Indexando manpages...")

def load_chunks(text, max_lines=40):
    lines = text.splitlines()
    return ['\n'.join(lines[i:i+max_lines]) for i in range(0, len(lines), max_lines)]

def index_manpage(path):
    try:
        with gzip.open(path, 'rt', errors='ignore') as f: text = f.read()
        file_hash = hashlib.sha256(text.encode()).hexdigest()
        chunks = load_chunks(text)
        embeddings = model.encode(chunks)
        ids = [f"man_{path.stem}_{i}" for i in range(len(chunks))]
        metadatas = [{
            'source': str(path),
            'chunk_index': i,
            'file_hash': file_hash,
            'type': 'manpage'
        } for i in range(len(chunks))]
        collection.upsert(documents=chunks, ids=ids, embeddings=embeddings.tolist(), metadatas=metadatas)
        print(f"📘 [MAN] {path} ({len(chunks)} blocos)")
    except Exception as e:
        print(f"⚠️ Erro em manpage {path}: {e}")

man_files = []
for sec in ['man1', 'man5', 'man8']:
    man_dir = man_root / sec
    if man_dir.exists():
        man_files += list(man_dir.glob('*.gz'))
        
with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
    executor.map(index_manpage, man_files)
print("✅ manpages concluído")
print(f"📦 Total de documentos no índice: {collection.count()}")
