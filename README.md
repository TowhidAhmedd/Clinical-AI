# 🏥 Clinical RAG Assistant

> A production-ready **Retrieval-Augmented Generation (RAG)** system for medical education, built with LangGraph multi-agent orchestration, Pinecone vector store, and Groq LLMs.

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)
![LangGraph](https://img.shields.io/badge/LangGraph-0.1-purple)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## ⚕️ Medical Safety Disclaimer

> **This system is for medical education only.** It cannot and must not be used for clinical diagnosis, treatment decisions, prescriptions, dosage guidance, or emergency medical advice. Always consult a qualified healthcare professional.

---

## 📋 Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [API Keys Setup](#api-keys-setup)
- [Local Development](#local-development)
- [Docker Deployment](#docker-deployment)
- [Render Deployment](#render-deployment)
- [API Reference](#api-reference)
- [Testing](#testing)
- [Project Structure](#project-structure)
- [Configuration](#configuration)

---

## ✨ Features

### RAG Pipeline
- **Multi-format ingestion** — PDF, DOCX, TXT
- **Local embeddings** — BAAI/bge-small-en-v1.5 (no external API)
- **Vector search** — Pinecone (production) / In-memory (dev)
- **Reranking** — FlashRank cross-encoder reranking
- **Context compression** — Token-budget-aware context window
- **Citation generation** — Document + page + chunk-level citations
- **Confidence scoring** — Weighted retrieval confidence per query

### Multi-Agent LangGraph Workflow
```
User Query
    │
    ▼
┌─────────────────┐
│  Query Router   │  ← Classifies: MEDICAL_EDUCATION | DOCUMENT_QUESTION
│     Agent       │               GENERAL_QUESTION | UNSAFE_MEDICAL_REQUEST
└────────┬────────┘
         │ (if safe)
    ▼
┌─────────────────┐
│  Safety Agent   │  ← LLM-based deep safety check (clinical advice detection)
└────────┬────────┘
         │ (if safe)
    ▼
┌─────────────────┐
│ Retrieval Agent │  ← Semantic search → Rerank → Compress → Cite
└────────┬────────┘
         │ (if sufficient context)
    ▼
┌─────────────────┐
│  Answer Agent   │  ← Grounded generation + hallucination guard + output guard
└────────┬────────┘
         │
    ▼
 Final Response (answer + citations + confidence + safety note)
```

### Safety Guardrails (4-layer)
| Layer | What it checks |
|-------|---------------|
| **Input Guardrail** | Prompt injection, jailbreaks, role manipulation, diagnosis/prescription/dosage requests (regex) |
| **Safety Agent** | LLM-based clinical advice detection (second pass) |
| **Retrieval Guardrail** | Checks chunks exist and have sufficient relevance score |
| **Output Guardrail** | Blocks generated diagnoses, dosage instructions, treatment plans |

### Security
- JWT authentication (login → Bearer token)
- API key header validation
- Rate limiting (slowapi)
- Input validation (Pydantic)
- File type and size validation (PDF/DOCX/TXT, 50 MB max)
- Non-root Docker user

### Observability
- Structured logging (loguru)
- LangSmith tracing (optional)
- Per-request trace IDs
- Timing instrumentation

---

## 🏗️ Architecture

```
clinical-rag-assistant/
├── backend/
│   ├── api/               # FastAPI route handlers + Pydantic schemas
│   ├── agents/            # LangGraph agent node functions
│   ├── graph/             # LangGraph workflow assembly & runner
│   ├── rag/               # Document processing, chunking, retrieval
│   ├── vectorstore/       # Pinecone + InMemory vector store
│   ├── embeddings/        # Local HuggingFace embedding wrapper
│   ├── llm/               # Groq LLM wrapper + system prompts
│   ├── guardrails/        # Medical safety guardrail functions
│   ├── security/          # JWT auth, API key, file validation
│   ├── observability/     # LangSmith config + RAG tracer
│   ├── utils/             # File utils, helpers
│   ├── config.py          # Pydantic settings (env-driven)
│   └── main.py            # FastAPI app entrypoint
├── frontend/
│   ├── components/        # API client
│   └── app.py             # Streamlit application
├── tests/                 # 98 tests across 5 test files
│   ├── conftest.py
│   ├── test_guardrails.py
│   ├── test_rag.py
│   ├── test_rag_eval.py
│   ├── test_agents.py
│   └── test_api.py
├── data/
│   ├── uploads/           # Uploaded document storage
│   └── processed/
├── Dockerfile             # Backend container
├── Dockerfile.frontend    # Frontend container
├── docker-compose.yml     # Local orchestration
├── render.yaml            # Render.com deployment
└── requirements.txt
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- [Groq API key](https://console.groq.com) (free)
- [Pinecone API key](https://app.pinecone.io) (free tier available)
- Optional: [LangSmith API key](https://smith.langchain.com)

### 1. Clone & Install

```bash
git clone <repo-url>
cd clinical-rag-assistant
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` with your keys:

```env
GROQ_API_KEY=gsk_...
PINECONE_API_KEY=...
PINECONE_ENVIRONMENT=us-east-1
PINECONE_INDEX_NAME=clinical-rag
JWT_SECRET_KEY=your-long-random-secret
API_KEY=your-api-key
```

### 3. Start Backend

```bash
cd clinical-rag-assistant
uvicorn backend.main:app --reload --port 8000
```

API docs: http://localhost:8000/docs

### 4. Start Frontend

```bash
BACKEND_URL=http://localhost:8000 streamlit run frontend/app.py
```

Frontend: http://localhost:8501

---

## 🔑 API Keys Setup

### Groq (Required — LLM)
1. Sign up at https://console.groq.com
2. Create an API key
3. Set `GROQ_API_KEY` in `.env`
4. Default model: `llama3-70b-8192` (also supports `llama3-8b-8192`, `mixtral-8x7b-32768`)

### Pinecone (Required for production — Vector DB)
1. Sign up at https://app.pinecone.io
2. Create a project → create an index:
   - **Name:** `clinical-rag` (or set `PINECONE_INDEX_NAME`)
   - **Dimensions:** `384`
   - **Metric:** `cosine`
3. Copy the API key → set `PINECONE_API_KEY`
4. Set `PINECONE_ENVIRONMENT` (e.g. `us-east-1`)

> **Dev mode:** If `PINECONE_API_KEY` is empty, the system uses an in-memory vector store automatically. Documents are lost on restart.

### LangSmith (Optional — Observability)
1. Sign up at https://smith.langchain.com
2. Create a project named `clinical-rag-assistant`
3. Set `LANGCHAIN_API_KEY`, `LANGCHAIN_TRACING_V2=true`

---

## 🐳 Docker Deployment

### Build & Run Locally

```bash
cp .env.example .env   # fill in your keys
docker compose up --build
```

- Backend: http://localhost:8000
- Frontend: http://localhost:8501

### Build Images Individually

```bash
# Backend
docker build -t clinical-rag-backend .

# Frontend
docker build -t clinical-rag-frontend -f Dockerfile.frontend .
```

---

## ☁️ Render Deployment

### Using render.yaml (Recommended)

```bash
# Install Render CLI
npm install -g @render-com/cli

# Deploy
render blueprint apply render.yaml
```

Set the following environment variables in the Render dashboard:
- `GROQ_API_KEY`
- `PINECONE_API_KEY`
- `LANGCHAIN_API_KEY` (optional)

`JWT_SECRET_KEY` and `API_KEY` are auto-generated by Render.

### Manual Deploy

1. Connect your GitHub repo to Render
2. **Backend:** New Web Service → Docker → set root dir `.` → Dockerfile `./Dockerfile`
3. **Frontend:** New Web Service → Docker → Dockerfile `./Dockerfile.frontend`
4. Set env vars for backend, set `BACKEND_URL` for frontend to backend's URL

---

## 📡 API Reference

### Authentication

```http
POST /auth/login
Content-Type: application/json

{"username": "admin", "password": "admin123"}
```

Response:
```json
{"access_token": "eyJ...", "token_type": "bearer", "expires_in": 3600}
```

### Chat (RAG Query)

```http
POST /chat/query
Authorization: Bearer <token>
Content-Type: application/json

{
  "query": "What is the mechanism of action of ACE inhibitors?",
  "doc_filter": null
}
```

Response:
```json
{
  "answer": "## Answer\nACE inhibitors...\n\n## Key Points\n...",
  "sources": [
    {
      "chunk_id": "abc-p1-c0-x1y2",
      "document_name": "cardiology.pdf",
      "page_number": 12,
      "score": 0.924,
      "excerpt": "ACE inhibitors are a class..."
    }
  ],
  "confidence": 0.891,
  "query_type": "MEDICAL_EDUCATION",
  "blocked": false,
  "safety_note": "Educational information only..."
}
```

### Document Upload

```http
POST /documents/upload
Authorization: Bearer <token>
Content-Type: multipart/form-data

file=<PDF/DOCX/TXT file>
```

### Document List

```http
GET /documents/list
Authorization: Bearer <token>
```

### Document Delete

```http
DELETE /documents/{doc_id}
Authorization: Bearer <token>
```

### Health Check

```http
GET /health
```

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage report
pytest tests/ --cov=backend --cov-report=term-missing

# Run specific test file
pytest tests/test_guardrails.py -v
pytest tests/test_agents.py -v
pytest tests/test_rag_eval.py -v

# Run only API tests
pytest tests/test_api.py -v
```

### Test Coverage Summary

| Test File | Description | Tests |
|-----------|-------------|-------|
| `test_guardrails.py` | Input/output/retrieval/hallucination guardrails | 23 |
| `test_rag.py` | Document processing, chunking, retrieval, citations | 18 |
| `test_rag_eval.py` | Context precision, recall, faithfulness, relevancy | 18 |
| `test_agents.py` | Router, safety, answer, final response agents | 20 |
| `test_api.py` | Auth, chat, document upload/list/delete endpoints | 19 |

**Total: 98 tests**

---

## ⚙️ Configuration

All configuration is in `.env`. Key variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | — | Groq API key (required) |
| `GROQ_MODEL` | `llama3-70b-8192` | LLM model name |
| `PINECONE_API_KEY` | — | Pinecone API key (optional, falls back to in-memory) |
| `PINECONE_INDEX_NAME` | `clinical-rag` | Pinecone index name |
| `PINECONE_ENVIRONMENT` | `us-east-1` | Pinecone region |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Local embedding model |
| `JWT_SECRET_KEY` | — | JWT signing key (required, use long random string) |
| `JWT_ACCESS_TOKEN_EXPIRE_MINUTES` | `60` | Token TTL |
| `API_KEY` | — | API key for service-to-service auth |
| `MAX_FILE_SIZE_MB` | `50` | Max upload file size |
| `LANGCHAIN_TRACING_V2` | `false` | Enable LangSmith tracing |
| `LANGCHAIN_API_KEY` | — | LangSmith API key |
| `APP_ENV` | `development` | Environment name |
| `LOG_LEVEL` | `INFO` | Logging level |

### Supported LLM Models (Groq)

| Model | ID | Context |
|-------|-----|---------|
| Llama 3 70B | `llama3-70b-8192` | 8K |
| Llama 3 8B | `llama3-8b-8192` | 8K |
| Mixtral 8x7B | `mixtral-8x7b-32768` | 32K |
| Gemma 7B | `gemma-7b-it` | 8K |

---

## 📝 Default Credentials

| Username | Password | Role |
|----------|----------|------|
| `admin` | `admin123` | Admin |
| `demo` | `demo123` | Demo user |

> **Production:** Replace the in-memory user store in `backend/security/auth.py` with a proper database (PostgreSQL + SQLAlchemy recommended).

---

## 🔒 Production Hardening Checklist

- [ ] Replace `FAKE_USERS_DB` with a real database
- [ ] Set strong, random `JWT_SECRET_KEY` (32+ chars)
- [ ] Set `APP_ENV=production`
- [ ] Configure Pinecone (don't rely on in-memory in production)
- [ ] Enable HTTPS (handled by Render/cloud provider)
- [ ] Set `ALLOWED_ORIGINS` to your actual frontend domain
- [ ] Enable LangSmith for observability
- [ ] Review and tighten CORS origins

---

## 📄 License

MIT License — see LICENSE for details.

---

*Built with ❤️ for medical education. Not a medical device.*
