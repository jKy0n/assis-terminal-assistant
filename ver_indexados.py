import chromadb

client = chromadb.HttpClient(host="localhost", port=8000)
col = client.get_or_create_collection("assis-docs")

results = col.get(include=["documents", "metadatas"])

for i in range(len(results["documents"])):
    doc = results["documents"][i]
    meta = results["metadatas"][i]
    print(f"🧾 Origem: {meta.get('source')} — Bloco {meta.get('chunk_index')}")
    print(doc)
    print("—" * 40)
