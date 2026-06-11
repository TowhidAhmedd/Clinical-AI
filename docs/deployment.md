# Deployment Guide

## Render.com (Recommended)

### Step-by-Step

1. **Push to GitHub**
   ```bash
   git init && git add . && git commit -m "Initial commit"
   git remote add origin <your-github-repo-url>
   git push -u origin main
   ```

2. **Create Render Web Services**

   **Backend:**
   - New → Web Service → Connect GitHub repo
   - Name: `clinical-rag-backend`
   - Runtime: Docker
   - Dockerfile Path: `./Dockerfile`
   - Instance Type: Starter ($7/mo) or higher
   - Health Check Path: `/health`

   **Frontend:**
   - New → Web Service → Connect GitHub repo
   - Name: `clinical-rag-frontend`
   - Runtime: Docker
   - Dockerfile Path: `./Dockerfile.frontend`

3. **Set Environment Variables (Backend)**

   In Render Dashboard → clinical-rag-backend → Environment:
   ```
   GROQ_API_KEY=gsk_...
   PINECONE_API_KEY=...
   PINECONE_ENVIRONMENT=us-east-1
   PINECONE_INDEX_NAME=clinical-rag
   JWT_SECRET_KEY=<generate: openssl rand -hex 32>
   API_KEY=<generate: openssl rand -hex 16>
   APP_ENV=production
   GROQ_MODEL=llama3-70b-8192
   EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
   ALLOWED_ORIGINS=https://your-frontend.onrender.com
   ```

4. **Set Environment Variables (Frontend)**
   ```
   BACKEND_URL=https://clinical-rag-backend.onrender.com
   ```

5. **Deploy** — Render auto-deploys on git push.

---

## Docker Compose (Local / Self-hosted)

```bash
cp .env.example .env   # Fill in API keys
docker compose up --build -d
```

- Backend: http://localhost:8000/docs
- Frontend: http://localhost:8501

---

## Important Notes

### Embedding Model Download
The Dockerfile pre-downloads `BAAI/bge-small-en-v1.5` during build. On Render free/starter plans, the build may time out if the model download is slow. If this happens:
- Use `sentence-transformers/all-MiniLM-L6-v2` (smaller) instead
- Or set `EMBEDDING_MODEL=mock` to use mock embeddings for testing

### Pinecone Index Setup
Before deploying, create your Pinecone index:
```
Name: clinical-rag (or your PINECONE_INDEX_NAME value)
Dimensions: 384
Metric: cosine
Cloud: AWS
Region: us-east-1 (match PINECONE_ENVIRONMENT)
```

### Cold Starts
Render free tier services spin down after inactivity. Backend cold start with model loading can take 60–90 seconds. Use Starter+ plan for always-on service.
