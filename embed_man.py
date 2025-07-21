# ~/.assis/embed_man.py (otimizado com multiprocess e batching)
import pathlib
import gzip
import hashlib
import chromadb
import concurrent.futures
from sentence_transformers import SentenceTransformer

# Initial Setup
MAX_THREADS = 8
MAX_SIZE = 1024 * 500  # manpages podem ser maiores
BATCH_SIZE = 256

model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.HttpClient(host="localhost", port=8000)
collection = chroma_client.get_or_create_collection(
  name="assis-docs",
  metadata={
    "hnsw:num_threads": 6,
    "hnsw:construction_ef": 128,
    "hnsw:search_ef": 64
  }
)

man_root = pathlib.Path('/usr/share/man')
man_files = []
for sec in ['man1', 'man5', 'man8']:
    man_dir = man_root / sec
    if man_dir.exists():
        man_files += list(man_dir.glob('*.gz'))

def load_chunks(text, max_lines=40):
    lines = text.splitlines()
    return ['\n'.join(lines[i:i+max_lines]) for i in range(0, len(lines), max_lines)]

def process_man_file(path):
    try:
        with gzip.open(path, 'rt', errors='ignore') as f:
            text = f.read()
        file_hash = hashlib.sha256(text.encode()).hexdigest()
        chunks = load_chunks(text)
        return [(f"man_{path.stem}_{i}", chunk, {
            'source': str(path),
            'chunk_index': i,
            'file_hash': file_hash,
            'type': 'manpage'
        }) for i, chunk in enumerate(chunks)]
    except Exception as e:
        print(f"⚠️ Erro em manpage {path}: {e}")
        return []

print("🔄 Indexando manpages...")
documents = []
with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
    for result in executor.map(process_man_file, man_files):
        if result:
            documents.extend(result)

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
        print(f"📘 [MAN] Lote {i // BATCH_SIZE + 1} de {len(documents) // BATCH_SIZE + 1} indexado")
    except Exception as e:
        print(f"❌ Erro no lote {i}: {e}")

print("✅ manpages concluído")
print(f"📦 Total de documentos no índice: {collection.count()}")
