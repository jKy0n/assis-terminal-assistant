# import os
# import hashlib
# import chromadb
# from sentence_transformers import SentenceTransformer

# # Diretórios e arquivos a serem indexados
# TARGET_FILES = [
#     "/etc/fstab",
#     "/etc/hostname",
#     "/etc/pacman.conf",
#     "/etc/resolv.conf",
#     "/etc/sudoers",
#     "/etc/systemd/system.conf",
#     # "/etc/NetworkManager/NetworkManager.conf",
# ]

# # Carrega modelo de embeddings local
# model = SentenceTransformer("all-MiniLM-L6-v2")

# # Conecta ao servidor ChromaDB
# chroma_client = chromadb.HttpClient(host="localhost", port=8000)
# collection = chroma_client.get_or_create_collection(name="assis-docs")

# def hash_file(path):
#     try:
#         with open(path, "rb") as f:
#             return hashlib.sha256(f.read()).hexdigest()
#     except:
#         return ""

# def load_chunks(path, max_lines=20):
#     try:
#         with open(path, "r") as f:
#             lines = f.readlines()
#         chunks = []
#         for i in range(0, len(lines), max_lines):
#             chunk = "".join(lines[i:i+max_lines])
#             chunks.append(chunk)
#         return chunks
#     except:
#         return []

# print("🔄 Iniciando indexação com ChromaDB...")
# doc_ids = []
# for path in TARGET_FILES:
#     chunks = load_chunks(path)
#     file_id = path.replace("/", "_")
#     file_hash = hash_file(path)

#     for idx, chunk in enumerate(chunks):
#         if not chunk.strip():
#             continue
#         doc_id = f"{file_id}_{idx}"
#         doc_ids.append(doc_id)

#         embedding = model.encode(chunk).tolist()

#         collection.upsert(
#             documents=[chunk],
#             ids=[doc_id],
#             embeddings=[embedding],
#             metadatas=[{
#                 "source": path,
#                 "chunk_index": idx,
#                 "file_hash": file_hash,
#             }]
#         )
#         print(f"📄 Indexado: {path} [bloco {idx}]")

# print("✅ Indexação concluída com sucesso!")
# print(f"📦 Total de documentos no índice: {collection.count()}")



import os
import pathlib
import gzip
import hashlib
import chromadb
import concurrent.futures
from sentence_transformers import SentenceTransformer

# Carrega modelo de embeddings local
model = SentenceTransformer("all-MiniLM-L6-v2")

# Conecta ao servidor ChromaDB
chroma_client = chromadb.HttpClient(host="localhost", port=8000)
collection = chroma_client.get_or_create_collection(name="assis-docs")

# Evita arquivos muito grandes ou binários
MAX_SIZE = 1024 * 100
MAX_THREADS = 8

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
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except:
        return ""

def load_chunks(text, max_lines=20):
    lines = text.splitlines()
    chunks = []
    for i in range(0, len(lines), max_lines):
        chunk = "\n".join(lines[i:i+max_lines])
        chunks.append(chunk)
    return chunks

def index_config_file(path):
    if not is_valid_file(path):
        return
    try:
        with open(path, "r", errors="ignore") as f:
            text = f.read()
        file_hash = hash_file(path)
        chunks = load_chunks(text)
        embeddings = model.encode(chunks)

        collection.upsert(
            documents=chunks,
            ids=[f"etc_{path.replace('/', '_')}_{i}" for i in range(len(chunks))],
            embeddings=embeddings,
            metadatas=[{
                "source": path,
                "chunk_index": i,
                "file_hash": file_hash,
                "type": "config"
            } for i in range(len(chunks))]
        )
        print(f"📄 [CONFIG] {path} ({len(chunks)} blocos)")
    except Exception as e:
        print(f"⚠️ Erro ao processar {path}: {e}")

def index_manpage(path):
    try:
        with gzip.open(path, "rt", errors="ignore") as f:
            text = f.read()
        file_hash = hashlib.sha256(text.encode()).hexdigest()
        chunks = load_chunks(text, max_lines=40)
        embeddings = model.encode(chunks)

        collection.upsert(
            documents=chunks,
            ids=[f"man_{path.stem}_{i}" for i in range(len(chunks))],
            embeddings=embeddings,
            metadatas=[{
                "source": str(path),
                "chunk_index": i,
                "file_hash": file_hash,
                "type": "manpage"
            } for i in range(len(chunks))]
        )
        print(f"📘 [MAN] {path} ({len(chunks)} blocos)")
    except Exception as e:
        print(f"⚠️ Erro ao processar manpage {path}: {e}")

print("🔄 Iniciando indexação com ChromaDB...")

# Arquivos alvo do /etc
CONFIG_FILES = [
    "/etc/fstab",
    "/etc/hostname",
    "/etc/pacman.conf",
    "/etc/resolv.conf",
    # "/etc/sudoers",
    "/etc/systemd/system.conf",
]

with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
    executor.map(index_config_file, CONFIG_FILES)

# Indexa páginas man
man_root = pathlib.Path("/usr/share/man")
man_files = [p for section in man_root.glob("man[1-9]*") for p in section.glob("*.gz")]

with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
    executor.map(index_manpage, man_files)

print("✅ Indexação concluída com sucesso!")
print(f"📦 Total de documentos no índice: {collection.count()}")
