# RAG — Retriever-Augmented Generation (SIH25231)

🚀 A local Retriever-Augmented Generation (RAG) project combining a React client and a Python server to index documents, create semantic embeddings, and answer queries using local LLMs and vector stores.

🔎 Quick overview

- **Client:** Vite + React frontend located in `client/` (chat UI, landing page, API service).
- **Server:** Python backend in `server/` (APIs, ingestion, retrieval, LLM wrappers).
- **Models:** Local model snapshots are stored under `models/` and `models_cache/`.
- **Data & DB:** Document uploads and vector DB files under `data/` and `server/data/vector_db/` (Chroma SQLite file included).

📁 Repo structure (high level)

- `client/` — Frontend app (Vite, React). Key files: `src/`, `package.json`, `vite.config.js`.
- `server/` — Backend service. Key files: `run.py`, `requirements.txt`, `api/`, `models/`, `database/`.
- `models/` & `models_cache/` — Local model snapshots and caches.
- `data/` — Uploaded documents and vector DB storage (`vector_db/chroma.sqlite3`).
- `processing/`, `retrieval/`, `utils/` — Document processing, hybrid search and helpers.

✨ Features

- Local-first RAG pipeline: ingest documents, create embeddings, and answer queries locally.
- Supports multiple local embeddings/LLM snapshots (see `models/`).
- React chat UI with file upload and message routing in `client/pages/chatPage.jsx`.
- Vector DB backed by Chroma (SQLite file included for quick start).

⚙️ Prerequisites

- Python 3.10+ (create a venv recommended)
- Node.js 16+ / npm or yarn
- Enough disk space for local models (varies by model)

💻 Local development — quick start

1. Backend (Windows PowerShell example)

```powershell
cd server
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python run.py
```

- The server exposes API endpoints defined in `server/api/` (check `run.py` and `server/api/main.py`).

2. Frontend

```bash
cd client
npm install
npm run dev
```

- Open the dev URL shown by Vite (usually http://localhost:5173).
- The frontend talks to the Python backend API (CORS/settings in `server/config/settings.py`).

📥 Adding documents & rebuilding embeddings

- Upload files using the frontend Upload flow or place documents into `server/data/uploads/` and run any provided ingestion scripts (see `processing/document_processor.py`).
- Rebuild the vector DB by running the ingestion/embedding script in `processing/` or the server's ingestion endpoint.

🧠 Models & embeddings

- Local model snapshots live in `models/` and `models_cache/`. Copy or download required model snapshots into these folders before starting if you want to use specific LLMs or embedding models.
- The repository includes references to models like `models--nomic-ai--nomic-embed-text-v1.5` and `models--microsoft--phi-1_5` — ensure you have licensing rights and required files.

🔧 Configuration

- Server settings: `server/config/settings.py`
- Vector DB path: `server/data/vector_db/chroma.sqlite3` (adjust in code if you move it)

🧪 Testing & troubleshooting

- Check `logs/` for runtime logs (both root `logs/` and `server/logs/`)
- If CPU/GPU resources are limited, prefer smaller embedding models or run embeddings on a subset of documents.

🤝 Contributing

- Want to improve the UI, add model adapters, or integrate a new vector DB? Fork, add a feature branch, and open a PR.
- Please include tests and update this README with any new setup details.

📜 License

- No license file included by default. Add a `LICENSE` file or let the maintainers know which license to apply.

❓ Need help?

- Tell me which part you want: run the app locally, connect a new model, or improve the UI. I can update README with more detailed commands or create setup scripts.

---

Made with ❤️ for the SIH25231 project — RAG local demo.
