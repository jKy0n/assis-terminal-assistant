import sys
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
import subprocess
import threading
import time
import shutil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("query", nargs="*", help="Pergunta para o assistente")
parser.add_argument("--file", help="Arquivo para incluir no contexto")
args = parser.parse_args()

query = " ".join(args.query)
if not query:
    print("❌ Nenhuma pergunta fornecida.")
    sys.exit(1)

extra_context = ""
if args.file:
    try:
        with open(args.file, "r", errors="ignore") as f:
            extra_context = f"\n### Conteúdo do arquivo {args.file}:\n{f.read()}\n"
    except Exception as e:
        print(f"⚠️ Não foi possível ler o arquivo: {e}")

# Conecta ao servidor ChromaDB
chroma_client = chromadb.HttpClient(host="localhost", port=8000)
collection = chroma_client.get_or_create_collection(name="assis-docs")

# Usa modelo de embedding local
model = SentenceTransformer("all-MiniLM-L6-v2")

# Faz embedding da pergunta
query_embedding = model.encode(query).tolist()

# Busca semântica
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=4
)

# Verifica se encontrou contexto
docs = results.get("documents", [[]])[0]
sources = results.get("metadatas", [[]])[0]

if docs or extra_context:
    context = ""
    if docs:
        for i, doc in enumerate(docs):
            origem = sources[i].get("source", "desconhecido")
            context += f"### Origem: {origem}\n{doc}\n\n"
    context += extra_context

    prompt = f"""Você é o Assis, um assistente virtual especializado em Linux.
Baseie sua resposta nas informações abaixo:

{context}

Agora responda à pergunta:
{query}
"""
else:
    print("⚠️ Nenhum contexto encontrado. Usando conhecimento interno do modelo.\n")
    prompt = query

# Spinner de carregamento
stop_spinner = False

def spinner():
    dots = ["", ".", "..", "..."]
    while not stop_spinner:
        for d in dots:
            if stop_spinner:
                break
            print(f"\r⏳ Gerando resposta{d}{' ' * 5}", end="", flush=True)
            time.sleep(0.5)
    print("\r", end="")  # limpa ao final

spinner_thread = threading.Thread(target=spinner)
spinner_thread.start()

# Chama o modelo com prompt direto
result = subprocess.run(
    ["ollama", "run", "mistral:7b-instruct-v0.3-q4_K_M"],
    input=prompt,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)

# Para o spinner
stop_spinner = True
spinner_thread.join()

# Exibe resultado formatado com glow, se disponível
has_glow = shutil.which("glow")

print("\n📥 RESPOSTA:\n")
if result.stdout.strip():
    if has_glow:
        subprocess.run(["glow", "-s", "dark"], input=result.stdout, text=True)
    else:
        print(result.stdout.strip())
else:
    print("⚠️ Resposta vazia")

if result.returncode != 0 and result.stderr.strip():
    print("\n⚠️ Erro ao rodar o modelo:")
    print(result.stderr.strip())