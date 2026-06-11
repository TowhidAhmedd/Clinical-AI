<div align="center">

<img src="https://img.shields.io/badge/Clinical%20RAG%20Assistant-v2.0-brightgreen?style=for-the-badge&logo=medrt&logoColor=white" alt="Clinical RAG Assistant"/>

<h1>🏥 Clinical RAG Assistant</h1>

<p><strong>A production-ready AI-powered medical education platform built with LangGraph multi-agent orchestration, Retrieval-Augmented Generation (RAG), and advanced safety guardrails.</strong></p>

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776ab?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.1-7C3AED?style=flat-square&logo=langchain&logoColor=white)](https://github.com/langchain-ai/langgraph)
[![LangSmith](https://img.shields.io/badge/LangSmith-0.8.11-7C3AED?style=flat-square&logo=langchain&logoColor=white)](https://smith.langchain.com/)
[![Groq](https://img.shields.io/badge/Groq-Llama%203-F97316?style=flat-square&logo=groq&logoColor=white)](https://console.groq.com/)
[![Pinecone](https://img.shields.io/badge/Pinecone-Vector%20DB-00B388?style=flat-square&logo=pinecone&logoColor=white)](https://www.pinecone.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-98%20passing-22c55e?style=flat-square&logo=pytest&logoColor=white)]()

<br/>

[**🚀 Quick Start**](#-quick-start) · [**📖 API Docs**](#-api-reference) · [**🐳 Docker**](#-docker-deployment) · [**☁️ Deploy**](#-render-deployment) · [**🧪 Testing**](#-testing)

</div>

---

> [!CAUTION]
> **Medical Safety Disclaimer:** This system is strictly for **medical education and research purposes only**. It **cannot** be used for clinical diagnosis, treatment decisions, prescriptions, dosage guidance, or emergency medical advice. Always consult a qualified healthcare professional for any medical concerns.

---
# Watch Demo Video
[![Watch Video](https://img.youtube.com/vi/JQbHalHyE8Q/maxresdefault.jpg)](https://youtu.be/JQbHalHyE8Q)
---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Multi-Agent Workflow](#-multi-agent-workflow)
- [Safety Guardrails](#-safety-guardrails)
- [Quick Start](#-quick-start)
- [API Keys Setup](#-api-keys-setup)
- [Local Development](#-local-development)
- [Docker Deployment](#-docker-deployment)
- [Render Deployment](#-render-deployment)
- [API Reference](#-api-reference)
- [Testing](#-testing)
- [Configuration](#-configuration)
- [Security](#-security)
- [Project Structure](#-project-structure)

---

## 🌟 Overview

The **Clinical RAG Assistant** is a full-stack AI application that enables medical students, researchers, and educators to query medical knowledge through two seamlessly integrated modes:

| Mode | Description |
|------|-------------|
| 🌐 **Web Search Mode** | Searches trusted medical websites (NIH, CDC, Mayo Clinic, etc.) when no documents are uploaded |
| 📄 **Document RAG Mode** | Answers from your uploaded PDFs/DOCX files with exact page citations |
| 🔀 **Hybrid Mode** | Combines document search with web supplementation for maximum coverage |

The system is built around a **4-agent LangGraph pipeline** with **4-layer safety guardrails** ensuring every response is safe, grounded, and educationally appropriate.

---

## ✨ Key Features

<table>
<tr>
<td width="50%">

### 🤖 AI & RAG
- Multi-agent LangGraph workflow orchestration
- Groq Llama 3 LLM (ultra-fast inference)
- Local embeddings — `BAAI/bge-small-en-v1.5` (no API cost)
- FlashRank cross-encoder reranking
- Token-budget-aware context compression
- Wikipedia API fallback for web search
- Confidence scoring per query

</td>
<td width="50%">

### 🛡️ Safety & Security
- 4-layer medical safety guardrail pipeline
- JWT Bearer token authentication
- API key header validation
- Rate limiting via SlowAPI
- Pydantic input validation
- File type & size validation (PDF/DOCX/TXT, 50 MB)
- Non-root Docker execution

</td>
</tr>
<tr>
<td width="50%">

### 📚 Document Processing
- Multi-format ingestion (PDF, DOCX, TXT)
- Semantic chunking with overlap
- Pinecone vector database (production)
- In-memory fallback (zero-config dev)
- Page-level citation generation
- Document-level filtering

</td>
<td width="50%">

### 📊 Observability
- Structured logging with Loguru
- LangSmith tracing integration (optional)
- Per-request trace IDs
- Timing instrumentation
- Live health monitoring dashboard
- Comprehensive test suite (98 tests)

</td>
</tr>
</table>

---

## 🏗️ System Architecture

```
clinical-rag-assistant/
├── backend/                    # FastAPI application
│   ├── api/                    # Route handlers & Pydantic schemas
│   │   ├── auth_routes.py      # JWT login/logout
│   │   ├── chat_routes.py      # RAG query endpoint
│   │   └── document_routes.py  # Upload/list/delete documents
│   ├── agents/                 # LangGraph agent node functions
│   │   └── workflow_agents.py  # Router, Safety, Retrieval, Answer agents
│   ├── graph/                  # LangGraph workflow assembly & runner
│   │   └── rag_workflow.py
│   ├── rag/                    # Document processing pipeline
│   │   ├── document_processor.py
│   │   ├── retriever.py
│   │   └── web_search.py       # Tavily + Wikipedia fallback
│   ├── vectorstore/            # Pinecone & InMemory adapters
│   ├── embeddings/             # HuggingFace embedding wrapper
│   ├── llm/                    # Groq LLM wrapper + system prompts
│   ├── guardrails/             # Medical safety guardrail functions
│   ├── security/               # JWT auth, API key, file validation
│   ├── observability/          # LangSmith config & RAG tracer
│   ├── config.py               # Pydantic settings (env-driven)
│   └── main.py                 # FastAPI application entrypoint
│
├── frontend/                   # Streamlit UI
│   ├── components/
│   │   └── api_client.py       # Backend HTTP client
│   └── app.py                  # Main Streamlit application
│
├── tests/                      # 98 tests across 5 test files
│   ├── conftest.py
│   ├── test_guardrails.py      # 23 tests
│   ├── test_rag.py             # 18 tests
│   ├── test_rag_eval.py        # 18 tests
│   ├── test_agents.py          # 20 tests
│   └── test_api.py             # 19 tests
│
├── docs/                       # Extended documentation
├── .streamlit/config.toml      # Streamlit theme configuration
├── pyrightconfig.json          # Pylance/Pyright type-check config
├── Dockerfile                  # Backend container
├── Dockerfile.frontend         # Frontend container
├── docker-compose.yml          # Local orchestration
├── render.yaml                 # Render.com deployment manifest
└── requirements.txt
```

---

## 🔄 Multi-Agent Workflow

```
                        ┌─────────────────────────────┐
   User Query ─────────►│     Query Router Agent       │
                        │  Classifies query type &     │
                        │  determines search mode:     │
                        │  WEB_ONLY │ HYBRID │ DOC_ONLY│
                        └──────────┬──────────────────┘
                                   │ (if SAFE)
                                   ▼
                        ┌─────────────────────────────┐
                        │      Safety Agent            │
                        │  LLM-based deep safety       │
                        │  check — detects clinical    │
                        │  advice disguised as edu.    │
                        └──────────┬──────────────────┘
                                   │ (if SAFE)
                                   ▼
                        ┌─────────────────────────────┐
                        │     Retrieval Agent          │
                        │  ① Pinecone semantic search  │
                        │  ② Tavily web search         │
                        │  ③ Wikipedia API fallback    │
                        │  → Rerank → Compress → Cite  │
                        └──────────┬──────────────────┘
                                   │ (if sufficient context)
                                   ▼
                        ┌─────────────────────────────┐
                        │      Answer Agent            │
                        │  Grounded generation +       │
                        │  hallucination guard +       │
                        │  output safety filter        │
                        └──────────┬──────────────────┘
                                   │
                                   ▼
              ┌────────────────────────────────────────────┐
              │           Final Response                    │
              │  answer + citations + confidence + mode    │
              │  + safety disclaimer                        │
              └────────────────────────────────────────────┘
```

---

## 🛡️ Safety Guardrails

The system employs **4 independent layers** of safety, ensuring no harmful medical advice is ever produced:

| Layer | Type | What It Blocks |
|-------|------|----------------|
| **Input Guardrail** | Regex patterns | Prompt injection, jailbreaks, role manipulation, diagnosis/prescription/dosage requests |
| **Safety Agent** | LLM-based | Clinical advice disguised as educational questions |
| **Retrieval Guardrail** | Threshold check | Responses with insufficient or low-relevance evidence |
| **Output Guardrail** | Regex patterns | Generated diagnoses, specific dosages, treatment plans, prescriptions |

Additionally, web search results are filtered through:
- ✅ **Domain whitelist** — Only NIH, CDC, WHO, Mayo Clinic, NHS, and similar trusted sources
- ✅ **Content sanitisation** — Dosage/prescription sentences stripped before reaching the LLM
- ✅ **Trust scoring** — Results ranked by source authority (`.gov`, `.edu` > general web)

---

## 🚀 Quick Start

### Prerequisites

| Requirement | Notes |
|-------------|-------|
| Python 3.11+ | [Download](https://www.python.org/downloads/) |
| Groq API Key | Free at [console.groq.com](https://console.groq.com) |
| Pinecone API Key | Free tier at [app.pinecone.io](https://app.pinecone.io) *(optional — falls back to in-memory)* |
| Tavily API Key | Free at [tavily.com](https://tavily.com) *(optional — falls back to Wikipedia)* |

### 1. Clone & Install

```bash
git clone https://github.com/TowhidAhmedd/Clinical-AI.git
cd Clinical-AI
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` with your API keys:

```env
# LLM — Required
GROQ_API_KEY=gsk_...
GROQ_MODEL=llama-3.1-8b-instant

# Web Search — Optional (falls back to Wikipedia if missing)
TAVILY_API_KEY=tvly-...

# Vector DB — Optional (falls back to in-memory if missing)
PINECONE_API_KEY=...
PINECONE_ENVIRONMENT=us-east-1
PINECONE_INDEX_NAME=clinical-rag

# Security — Required for production
JWT_SECRET_KEY=your-long-random-secret-key-min-32-chars
API_KEY=your-internal-api-key
```

### 3. Start Backend

```bash
uvicorn backend.main:app --reload --port 8000
```

- **API:** http://localhost:8000
- **Interactive Docs (Swagger):** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

### 4. Start Frontend

```bash
streamlit run frontend/app.py
```

- **UI:** http://localhost:8501

### 5. Login & Start Asking

| Username | Password | Role |
|----------|----------|------|
| `admin` | `admin123` | Admin |
| `demo` | `demo123` | Demo user |

> [!WARNING]
> Change default credentials before any production deployment. Replace the in-memory user store in `backend/security/auth.py` with a proper database.

---

## 🔑 API Keys Setup

### Groq (Required — LLM Provider)

1. Sign up at [console.groq.com](https://console.groq.com)
2. Navigate to **API Keys** → **Create API Key**
3. Set `GROQ_API_KEY` in `.env`

**Supported models:**

| Model | ID | Best For |
|-------|----|----------|
| Llama 3.1 8B | `llama-3.1-8b-instant` | Fast, lightweight (recommended for dev) |
| Llama 3.3 70B | `llama-3.3-70b-versatile` | Best quality (recommended for production) |
| Llama 3 70B | `llama3-70b-8192` | Stable, well-tested |
| Mixtral 8x7B | `mixtral-8x7b-32768` | Long context window (32K) |

### Pinecone (Optional — Vector Database)

1. Sign up at [app.pinecone.io](https://app.pinecone.io)
2. Create a new index:
   - **Name:** `clinical-rag`
   - **Dimensions:** `384`
   - **Metric:** `cosine`
3. Copy your API key → set `PINECONE_API_KEY`

> [!NOTE]
> If `PINECONE_API_KEY` is not set, the system automatically uses an **in-memory vector store**. This is perfect for development but documents are lost on server restart.

### Tavily (Optional — Web Search)

1. Sign up at [tavily.com](https://tavily.com)
2. Get your API key → set `TAVILY_API_KEY`

> [!NOTE]
> If Tavily is not configured, the system falls back to **DuckDuckGo scraping**, then **Wikipedia API**. Web search will still work but may be slower.

### LangSmith (Optional — Observability)

1. Sign up at [smith.langchain.com](https://smith.langchain.com)
2. Create a project: `clinical-rag-assistant`
3. Set `LANGCHAIN_API_KEY=...` and `LANGCHAIN_TRACING_V2=true`

---

## 🐳 Docker Deployment

### Option 1: Docker Compose (Recommended)

```bash
# Clone and configure
git clone https://github.com/TowhidAhmedd/Clinical-AI.git
cd Clinical-AI
cp .env.example .env
# Edit .env with your API keys

# Build and start all services
docker compose up --build

# Run in background
docker compose up --build -d
```

| Service | URL |
|---------|-----|
| Backend API | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |
| Frontend UI | http://localhost:8501 |

```bash
# Stop services
docker compose down

# View logs
docker compose logs -f backend
docker compose logs -f frontend
```

### Option 2: Individual Containers

```bash
# Build backend
docker build -t clinical-rag-backend .

# Build frontend
docker build -t clinical-rag-frontend -f Dockerfile.frontend .

# Run backend
docker run -d --env-file .env -p 8000:8000 clinical-rag-backend

# Run frontend
docker run -d -e BACKEND_URL=http://localhost:8000 -p 8501:8501 clinical-rag-frontend
```

---

## ☁️ Render Deployment

### One-Click Deploy with render.yaml

1. Fork this repository on GitHub
2. Connect to [Render](https://render.com)
3. Select **New → Blueprint** → connect your forked repo
4. Render will auto-detect `render.yaml` and create both services

**Set the following environment variables in the Render dashboard:**

| Variable | Required |
|----------|----------|
| `GROQ_API_KEY` | ✅ Yes |
| `PINECONE_API_KEY` | ⚠️ Recommended |
| `TAVILY_API_KEY` | ⚠️ Recommended |
| `LANGCHAIN_API_KEY` | ❌ Optional |

> `JWT_SECRET_KEY` and `API_KEY` are auto-generated by Render's secret management.

### Manual Deploy

1. Connect your GitHub repo at [render.com](https://render.com)
2. **Backend:** New Web Service → Docker → Dockerfile: `./Dockerfile`
3. **Frontend:** New Web Service → Docker → Dockerfile: `./Dockerfile.frontend`
4. Set `BACKEND_URL` on the frontend service to the backend's Render URL

---

## 📡 API Reference

### Base URL

```
http://localhost:8000     # Local development
https://your-app.onrender.com  # Production
```

### Authentication

```http
POST /auth/login
Content-Type: application/json

{"username": "admin", "password": "admin123"}
```

```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

### Chat / RAG Query

```http
POST /chat/query
Authorization: Bearer <access_token>
Content-Type: application/json

{
  "query": "What is the mechanism of action of ACE inhibitors?",
  "doc_filter": null
}
```

```json
{
  "answer": "## ACE Inhibitors — Mechanism of Action\n\nACE inhibitors work by...",
  "sources": [
    {
      "chunk_id": "doc-p12-c0",
      "document_name": "cardiology_notes.pdf",
      "page_number": 12,
      "score": 0.924,
      "excerpt": "ACE inhibitors block the angiotensin-converting enzyme...",
      "url": null,
      "source_type": "document"
    }
  ],
  "confidence": 0.891,
  "query_type": "MEDICAL_EDUCATION",
  "search_mode": "HYBRID",
  "blocked": false,
  "safety_note": "Educational information only. Not medical advice."
}
```

### Document Endpoints

```http
# Upload document
POST /documents/upload
Authorization: Bearer <token>
Content-Type: multipart/form-data
file=<PDF|DOCX|TXT>

# List all documents
GET /documents/list
Authorization: Bearer <token>

# Delete a document
DELETE /documents/{doc_id}
Authorization: Bearer <token>
```

### Health Check

```http
GET /health
```

```json
{
  "status": "ok",
  "llm": "llama-3.1-8b-instant",
  "vector_store": "Pinecone",
  "web_search": "Tavily",
  "documents_indexed": 3
}
```

> [!TIP]
> The full interactive API documentation is available at **http://localhost:8000/docs** (Swagger UI) and **http://localhost:8000/redoc** (ReDoc) when the backend is running.

---

## 🧪 Testing

```bash
# Activate virtual environment first
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux

# Run all 98 tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=backend --cov-report=term-missing

# Run specific test modules
pytest tests/test_guardrails.py -v   # Safety guardrails
pytest tests/test_agents.py -v       # LangGraph agents
pytest tests/test_rag.py -v          # RAG pipeline
pytest tests/test_rag_eval.py -v     # RAG evaluation metrics
pytest tests/test_api.py -v          # REST API endpoints
```

### Test Coverage

| Test File | Coverage Area | Tests |
|-----------|---------------|:-----:|
| `test_guardrails.py` | Input/output/retrieval/hallucination guardrails | **23** |
| `test_rag.py` | Document processing, chunking, retrieval, citations | **18** |
| `test_rag_eval.py` | Context precision, recall, faithfulness, relevancy | **18** |
| `test_agents.py` | Router, safety, retrieval, answer agents | **20** |
| `test_api.py` | Auth, chat, document upload/list/delete endpoints | **19** |
| | **Total** | **98** |

---

## ⚙️ Configuration

All settings are driven by environment variables (`.env` file). No code changes required for deployment configuration.

| Variable | Default | Required | Description |
|----------|---------|:--------:|-------------|
| `GROQ_API_KEY` | — | ✅ | Groq LLM API key |
| `GROQ_MODEL` | `llama3-70b-8192` | — | Groq model identifier |
| `TAVILY_API_KEY` | — | ⚠️ | Tavily web search API key |
| `PINECONE_API_KEY` | — | ⚠️ | Pinecone vector DB key (in-memory fallback if empty) |
| `PINECONE_INDEX_NAME` | `clinical-rag` | — | Pinecone index name |
| `PINECONE_ENVIRONMENT` | `us-east-1` | — | Pinecone region |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | — | Local HuggingFace embedding model |
| `JWT_SECRET_KEY` | — | ✅ | JWT signing key (min. 32 chars) |
| `JWT_ACCESS_TOKEN_EXPIRE_MINUTES` | `60` | — | Token expiry in minutes |
| `API_KEY` | — | ✅ | Internal service-to-service API key |
| `MAX_FILE_SIZE_MB` | `50` | — | Maximum document upload size |
| `ALLOWED_ORIGINS` | `http://localhost:8501` | — | CORS allowed origins |
| `LANGCHAIN_TRACING_V2` | `false` | — | Enable LangSmith tracing |
| `LANGCHAIN_API_KEY` | — | ❌ | LangSmith API key |
| `APP_ENV` | `development` | — | Environment (`development`/`production`) |
| `LOG_LEVEL` | `INFO` | — | Logging verbosity |

---

## 🔒 Security

### Authentication Flow

```
Client ─── POST /auth/login ──► Backend
              username + password        │
                                         ├─ Verify credentials
                                         ├─ Generate JWT (HS256)
Client ◄── Bearer token ─────────────── │

Client ─── POST /chat/query ──► Middleware
              Authorization: Bearer ...  │
                                         ├─ Validate JWT signature
                                         ├─ Check expiry
                                         └─ Forward to route handler
```

### Production Hardening Checklist

- [ ] Replace `FAKE_USERS_DB` with PostgreSQL + SQLAlchemy
- [ ] Set strong, random `JWT_SECRET_KEY` (32+ chars)
- [ ] Set `APP_ENV=production`
- [ ] Configure Pinecone (never use in-memory in production)
- [ ] Enable HTTPS (handled by cloud provider / reverse proxy)
- [ ] Restrict `ALLOWED_ORIGINS` to your actual frontend domain
- [ ] Enable LangSmith for full observability
- [ ] Review and tighten CORS configuration
- [ ] Set up log aggregation (CloudWatch / Papertrail / Datadog)

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

Please ensure all tests pass before submitting: `pytest tests/ -v`

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Towhid Ahmed**

[![GitHub](https://img.shields.io/badge/GitHub-TowhidAhmedd-181717?style=flat-square&logo=github)](https://github.com/TowhidAhmedd/Clinical-AI)

---

<div align="center">

**Built with ❤️ for medical education.**

*This is not a medical device. Always consult a qualified healthcare professional.*

⭐ **Star this repo if you found it useful!** ⭐

</div>
