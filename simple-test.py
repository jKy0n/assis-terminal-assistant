import subprocess

prompt = "Qual é a capital do Brasil?"

result = subprocess.run(
    ["ollama", "run", "mistral:7b-instruct-v0.3-q4_K_M"],
    input=prompt,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)

print("📥 RESPOSTA:\n", result.stdout.strip() or "⚠️ NADA")

if result.stderr:
    print("⚠️ STDERR:\n", result.stderr.strip())