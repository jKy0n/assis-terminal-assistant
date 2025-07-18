# Assis — Terminal Virtual Assistant 🧠🐚

**Assis** is a lightweight terminal-based virtual assistant designed to help with Linux system administration tasks using local AI models, semantic search (RAG), and friendly Zsh integration.

## ✨ Features
- 🧠 Powered by [`mistral:7b-instruct-v0.3-q4_K_M`] via [Ollama](https://ollama.com)
- 🔍 Local RAG (retrieval-augmented generation) using [ChromaDB](https://www.trychroma.com/) + sentence-transformers (`all-MiniLM-L6-v2`)
- 📁 Indexed system files from `/etc` (e.g. `fstab`, `hostname`, `pacman.conf`, etc.)
- 💬 Zsh integration as a simple `assis` function
- 🎨 Output rendered with [Glow](https://github.com/charmbracelet/glow) for pretty Markdown in the terminal
- ⚡ Fast response times (CPU-only, optimized for real-world performance)

## 🧱 Tech stack
- Language: **Python 3.13**
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`
- Vector store: **ChromaDB** (via persistent HTTP server)
- Terminal UI: **Glow** for styled Markdown output
- AI backend: **Ollama** running **Mistral 7B instruct Q4_K_M** locally

## 🗂 File structure
```
~/.assis/
├── assis-query.py       # Main assistant logic (query + prompt + ollama)
├── embed.py             # Indexes target files into ChromaDB
├── assis.zsh            # Zsh function wrapper (CLI entry point)
├── ver_indexados.py     # Tool to inspect stored vector documents
├── simple-test.py       # Minimal prompt test
├── chroma/              # ChromaDB persistent vector store (excluded)
└── .gitignore           # Prevents DB and cache from polluting Git
```

## 🚀 Setup
1. Install dependencies:
   ```bash
   pip install chromadb sentence-transformers
   paru -S ollama glow
   ```
2. Start ChromaDB server:
   ```bash
   chroma run --host 127.0.0.1 --port 8000 --path ~/.assis/chroma &
   ```
3. Index system files:
   ```bash
   python ~/.assis/embed.py
   ```
4. Query Assis:
   ```bash
   python ~/.assis/assis-query.py "Explique a opção relatime no fstab"
   ```

## ⚙️ Coming soon
- Support for `/home/*/.config` files
- More advanced RAG rules & metadata
- Background re-indexing via systemd timer
- Multi-turn chat history support

## 📸 Screenshots
*Coming soon – will include sample terminal outputs with glow-rendered Markdown.*

---

Made with 💻 by jKyon (John K. L. Segundo) and ChatGPT
