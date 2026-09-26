# 🚀 Talk2Code

### Talk to your codebase.

Talk2Code is an open-source **AI developer agent** that helps developers understand and work with their codebase using natural language.

Connect a GitHub repository → Talk2Code indexes the code → ask questions → RAG retrieves relevant context → the LLM generates grounded answers with **source citations**.

## ✨ Features

- 🔗 GitHub repository integration
- 🧠 Codebase-aware RAG
- 🔎 Semantic code search
- 📚 Source citations
- 💾 Project & conversation memory
- 🤖 AI agent with tool calling
- 🎙️ Voice interaction
- 🛠️ Git, terminal & testing tools
- ✏️ Code modification

## 🏗️ Architecture

```text
GitHub Repo
     ↓
Index → Chunk → Embed → Vector DB
     ↓
   User Query
     ↓
  RAG Retrieval
     ↓
      LLM
     ↓
Answer + Citations

```

## 🛠️ Tech Stack

**Python · PyTorch · Transformers · LLM APIs · RAG · Embeddings · Vector DB · FastAPI · PostgreSQL/SQLite · Docker · MCP**

## 🤝 Contributing

Talk2Code is open source and contributions are welcome.

**Fork → Branch → Build → Test → Pull Request**

## 🎯 Vision

We're not building another generic chatbot.

**We're building an AI that understands your codebase.**

> **Talk to your codebase.** 🚀
