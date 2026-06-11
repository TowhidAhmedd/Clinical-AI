"""
Clinical RAG Assistant — Streamlit Frontend v2.0
No document upload needed — web search answers any medical education question.
Documents optional — upload to get document-grounded answers with citations.
"""
import streamlit as st
import os, sys, requests, time
from pathlib import Path

st.set_page_config(
    page_title="Clinical RAG Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "Clinical RAG Assistant v2.0 · Medical Education AI"},
)

sys.path.insert(0, str(Path(__file__).parent))
from components.api_client import APIClient

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# ── Session state ─────────────────────────────────────────────────
for k, v in {
    "client":        APIClient(BACKEND_URL),
    "authenticated": False,
    "username":      "",
    "chat_history":  [],
    "documents":     [],
    "flash":         None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── CSS ───────────────────────────────────────────────────────────
st.markdown("""<style>
/* ── Main background ── */
[data-testid="stAppViewContainer"]{background:#0a1f0e}
[data-testid="stMain"]{background:#0a1f0e}

/* ── Sidebar ── */
[data-testid="stSidebar"]{background:#0d2611}
[data-testid="stSidebar"] *{color:#b7e4c7!important}
[data-testid="stSidebar"] .stButton>button{
    background:#1a6b34;border:none;color:#fff!important;
    border-radius:8px;width:100%;margin:2px 0}
[data-testid="stSidebar"] .stButton>button:hover{background:#145228}

/* ── Header gradient ── */
.rag-header{
    background:linear-gradient(135deg,#0d2611 0%,#1a6b34 100%);
    padding:1.3rem 1.8rem;border-radius:14px;color:#e2f5e8;
    margin-bottom:1.2rem;box-shadow:0 4px 16px rgba(0,0,0,.35)}
.rag-header h1,.rag-header h2{margin:0 0 .25rem;color:#d4edda}
.rag-header p{margin:0;opacity:.85;font-size:.92rem;color:#b7e4c7}

/* ── Mode badges ── */
.mode-badge{
    display:inline-block;padding:.18rem .7rem;
    border-radius:99px;font-size:.76rem;font-weight:700;
    margin-left:.5rem;vertical-align:middle}
.mode-web{background:#bbf7d0;color:#065f46}
.mode-doc{background:#a7f3d0;color:#064e3b}
.mode-hybrid{background:#d1fae5;color:#047857}

/* ── Source cards ── */
.source-web{
    background:#132d18;border-left:4px solid #22c55e;
    border-radius:6px;padding:.6rem .9rem;margin:.3rem 0;
    font-size:.82rem;color:#b7e4c7}
.source-doc{
    background:#0d2611;border-left:4px solid #4ade80;
    border-radius:6px;padding:.6rem .9rem;margin:.3rem 0;
    font-size:.82rem;color:#b7e4c7}

/* ── Blocked / safety / tip boxes ── */
.blocked-box{
    background:#1a0a0a;border:1px solid #f87171;
    border-radius:8px;padding:.8rem;margin-top:.4rem;
    font-size:.88rem;color:#fca5a5}
.safety-note{
    background:#1a1a00;border:1px solid #fcd34d;
    border-radius:8px;padding:.5rem .9rem;font-size:.8rem;
    margin-top:.4rem;line-height:1.5;color:#fef08a}
.tip-box{
    background:#132d18;border:1px solid #22c55e;
    border-radius:10px;padding:.9rem 1.1rem;
    font-size:.88rem;line-height:1.6;margin-bottom:.8rem;color:#b7e4c7}

/* ── General text ── */
p, li, span, label, div {color:#e2f5e8}
h1,h2,h3,h4 {color:#d4edda}
code {background:#132d18!important;color:#4ade80!important}
</style>""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────
def refresh_docs():
    try:
        st.session_state.documents = st.session_state.client.list_documents()
    except Exception:
        pass

def mode_badge(mode: str) -> str:
    m = (mode or "").upper()
    labels = {"WEB_ONLY": ("🌐 Web Search","mode-web"),
              "DOC_ONLY": ("📄 Document","mode-doc"),
              "HYBRID":   ("🔀 Hybrid","mode-hybrid")}
    label, cls = labels.get(m, (m, "mode-web"))
    return f'<span class="mode-badge {cls}">{label}</span>'

def conf_color(c: float) -> str:
    if c >= 0.7: return "🟢"
    if c >= 0.4: return "🟡"
    return "🔴"


# ── LOGIN ─────────────────────────────────────────────────────────
def show_login():
    _, col, _ = st.columns([1,2,1])
    with col:
        st.markdown("""
        <div class="rag-header" style="text-align:center">
            <h1>🏥 Clinical RAG Assistant</h1>
            <p>Medical Education AI · Web Search + Document RAG</p>
        </div>""", unsafe_allow_html=True)

        # Quick backend check
        try:
            st.session_state.client.health()
            st.success("🟢 Backend connected")
        except Exception:
            st.warning(
                f"⏳ **Backend unreachable** at `{BACKEND_URL}`\n\n"
                "Run: `uvicorn backend.main:app --reload --port 8000`")
            if st.button("🔄 Retry"):
                st.rerun()
            return

        with st.form("login_form"):
            u = st.text_input("👤 Username", placeholder="admin  or  demo")
            p = st.text_input("🔒 Password", type="password")
            ok = st.form_submit_button("Sign In →", use_container_width=True, type="primary")

        if ok:
            if not u.strip() or not p.strip():
                st.error("Fill in both fields.")
                return
            with st.spinner("Authenticating…"):
                try:
                    st.session_state.client.login(u.strip(), p.strip())
                    st.session_state.authenticated = True
                    st.session_state.username = u.strip()
                    refresh_docs()
                    st.rerun()
                except requests.exceptions.ReadTimeout:
                    st.error("⏳ Timed out — backend is initialising. Wait 15s and retry.")
                except requests.exceptions.ConnectionError:
                    st.error(f"❌ Cannot connect to `{BACKEND_URL}`.")
                except Exception as e:
                    msg = str(e)
                    st.error("❌ Wrong credentials." if any(
                        x in msg for x in ["401","Unauthorized","Incorrect"]) else f"❌ {msg}")

        st.markdown("---")
        st.markdown("""
| Username | Password |
|----------|----------|
| `admin`  | `admin123` |
| `demo`   | `demo123`  |
""")
        st.markdown("""<div class="safety-note">
⚕️ <strong>Medical Safety Notice:</strong> This assistant provides
<em>educational information only</em>. It cannot diagnose conditions,
recommend medications, or replace professional medical advice.
</div>""", unsafe_allow_html=True)


# ── SIDEBAR ───────────────────────────────────────────────────────
def show_sidebar() -> str:
    with st.sidebar:
        st.markdown(f"### 👤 {st.session_state.username}")
        if st.button("🚪 Logout", use_container_width=True):
            for k,v in {"authenticated":False,"username":"",
                        "chat_history":[],"documents":[]}.items():
                st.session_state[k] = v
            st.session_state.client = APIClient(BACKEND_URL)
            st.rerun()

        st.markdown("---")
        page = st.radio("Nav", ["💬 Chat","📄 Documents","ℹ️ About"],
                        label_visibility="collapsed")

        st.markdown("---")
        try:
            h = st.session_state.client.health()
            st.success("🟢 API Online")
            with st.expander("⚙️ System"):
                for k,v in h.items():
                    st.markdown(f"**{k.replace('_',' ').title()}:** `{v}`")
        except requests.exceptions.ConnectionError:
            st.error("🔴 API Offline")
        except Exception:
            st.warning("🟡 API Slow")

        st.markdown("---")
        n = len(st.session_state.documents)
        st.markdown(f"📚 **{n}** doc{'s' if n!=1 else ''} indexed")
        st.markdown("""
<div style="font-size:.7rem;opacity:.5;margin-top:.8rem;line-height:1.7">
⚕️ Educational use only.<br>
Not a substitute for<br>professional medical advice.
</div>""", unsafe_allow_html=True)

    return page.split(" ",1)[1]


# ── CHAT ──────────────────────────────────────────────────────────
def show_chat_page():
    st.markdown("""
    <div class="rag-header">
        <h2>💬 Clinical Education Chat</h2>
        <p>Ask any medical education question — no document upload needed</p>
    </div>""", unsafe_allow_html=True)

    # ── How it works tip ──────────────────────────────────────────
    with st.expander("💡 How this works", expanded=len(st.session_state.chat_history) == 0):
        n = len(st.session_state.documents)
        if n == 0:
            st.markdown("""
<div class="tip-box">
<strong>🌐 Web Search Mode (active now)</strong><br>
You haven't uploaded any documents — that's completely fine!<br>
Ask any medical education question and the assistant will search
trusted medical websites (via <strong>Tavily</strong>) to answer you.<br><br>
<strong>Examples you can ask right now:</strong><br>
• What is the mechanism of action of ACE inhibitors?<br>
• Explain the pathophysiology of type 2 diabetes<br>
• What are the chambers of the human heart?<br>
• How does the renal system regulate blood pressure?<br><br>
📄 <em>Optionally, go to <strong>Documents</strong> tab to upload your own medical PDFs/notes
for document-grounded answers with page citations.</em>
</div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
<div class="tip-box">
<strong>🔀 Hybrid Mode (active now)</strong><br>
You have <strong>{n} document(s)</strong> indexed.
The assistant will first search your documents, then supplement with web search if needed.<br><br>
• Use the <strong>filter dropdown</strong> to restrict answers to one document<br>
• Leave filter on <em>All Documents</em> for hybrid web+doc answers
</div>""", unsafe_allow_html=True)

    # ── Document filter ───────────────────────────────────────────
    doc_filter = None
    if st.session_state.documents:
        opts = {"🗂 All Documents (Hybrid)": None}
        for d in st.session_state.documents:
            opts[f"📄 {d['filename']}"] = d["doc_id"]
        _, fc = st.columns([4,2])
        with fc:
            chosen = st.selectbox("Source filter", list(opts.keys()),
                                  label_visibility="collapsed")
            doc_filter = opts[chosen]

    st.markdown("---")

    # ── Chat history ──────────────────────────────────────────────
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"],
                             avatar="👤" if msg["role"]=="user" else "🏥"):
            # Mode badge next to assistant messages
            if msg["role"] == "assistant" and msg.get("search_mode"):
                st.markdown(mode_badge(msg["search_mode"]), unsafe_allow_html=True)

            st.markdown(msg["content"])

            # Sources
            if msg.get("sources"):
                conf  = msg.get("confidence", 0.0)
                icon  = conf_color(conf)
                label = f"📚 {len(msg['sources'])} source(s) · {icon} {conf:.0%} confidence"
                with st.expander(label, expanded=False):
                    for i, src in enumerate(msg["sources"], 1):
                        stype = src.get("source_type","document")
                        css   = "source-web" if stype=="web" else "source-doc"
                        icon2 = "🌐" if stype=="web" else "📄"
                        url_html = (f'<br><a href="{src["url"]}" target="_blank">'
                                    f'🔗 {src["url"][:70]}…</a>'
                                    if src.get("url") else "")
                        pg = f"· Page {src['page_number']}" if src.get("page_number") else ""
                        st.markdown(f"""
<div class="{css}">
<strong>{icon2} [{i}] {src['document_name']}</strong> {pg}
· Score <code>{src['score']:.3f}</code>{url_html}<br>
<em>{src['excerpt'][:240]}{"…" if len(src['excerpt'])>=240 else ""}</em>
</div>""", unsafe_allow_html=True)

            # Blocked
            if msg.get("blocked"):
                st.markdown(f"""
<div class="blocked-box">🚫 <strong>Blocked:</strong>
{msg.get('blocked_by','safety guardrails')}</div>""",
                            unsafe_allow_html=True)

    # ── Input ─────────────────────────────────────────────────────
    placeholder = ("Ask a medical education question… (web search active — no upload needed)"
                   if not st.session_state.documents
                   else "Ask about your documents or any medical topic…")

    if query := st.chat_input(placeholder):
        st.session_state.chat_history.append({"role":"user","content":query})

        mode_hint = ("🌐 Searching the web…" if not st.session_state.documents
                     else "🔍 Searching documents + web…")
        with st.spinner(mode_hint):
            try:
                resp = st.session_state.client.chat(query, doc_filter=doc_filter)
                st.session_state.chat_history.append({
                    "role":        "assistant",
                    "content":     resp["answer"],
                    "sources":     resp.get("sources", []),
                    "confidence":  resp.get("confidence", 0.0),
                    "search_mode": resp.get("search_mode"),
                    "blocked":     resp.get("blocked", False),
                    "blocked_by":  resp.get("blocked_by"),
                })
            except requests.exceptions.ReadTimeout:
                st.session_state.chat_history.append({
                    "role":"assistant",
                    "content":"⏳ **Timed out.** First query loads the embedding model. Try again.",
                })
            except requests.exceptions.ConnectionError:
                st.session_state.chat_history.append({
                    "role":"assistant",
                    "content":f"❌ Cannot reach `{BACKEND_URL}`. Is the backend running?",
                })
            except Exception as exc:
                st.session_state.chat_history.append({
                    "role":"assistant","content":f"❌ Error: {exc}",
                })
        st.rerun()

    if st.session_state.chat_history:
        if st.button("🗑️ Clear chat", type="secondary"):
            st.session_state.chat_history = []
            st.rerun()


# ── DOCUMENTS ─────────────────────────────────────────────────────
def show_documents_page():
    st.markdown("""
    <div class="rag-header">
        <h2>📄 Documents <small style="opacity:.7;font-size:.7em">(optional)</small></h2>
        <p>Upload your own medical notes, textbooks, or research papers for document-grounded answers</p>
    </div>""", unsafe_allow_html=True)

    # Flash message
    if st.session_state.flash:
        st.success(st.session_state.flash)
        st.session_state.flash = None

    # Why upload?
    st.info(
        "💡 **Document upload is optional.** The assistant answers any question via web search.\n\n"
        "Upload documents when you want answers grounded in **specific materials** — "
        "your lecture notes, a clinical guideline PDF, or a textbook chapter — "
        "with exact **page-level citations**.",
        icon="📎",
    )

    st.markdown("### ⬆️ Upload Document")
    uploaded = st.file_uploader(
        "PDF, DOCX, or TXT — max 50 MB",
        type=["pdf","docx","txt"],
        label_visibility="collapsed",
    )
    if uploaded:
        kb = len(uploaded.getvalue())/1024
        st.info(f"📄 **{uploaded.name}** · {'%.1f KB' % kb if kb<1024 else '%.2f MB' % (kb/1024)}")
        if st.button("🚀 Process & Index", type="primary"):
            with st.spinner(f"Chunking and embedding **{uploaded.name}**…"):
                try:
                    r = st.session_state.client.upload_document(
                        uploaded.getvalue(), uploaded.name)
                    st.session_state.flash = (
                        f"✅ **{uploaded.name}** indexed — {r['total_chunks']} chunks.")
                    refresh_docs()
                    st.rerun()
                except requests.exceptions.ReadTimeout:
                    st.error("⏳ Upload timed out. The embedding model is loading — retry in 30s.")
                except Exception as exc:
                    st.error(f"❌ Upload failed: {exc}")

    st.markdown("---")
    st.markdown("### 📚 Indexed Documents")
    refresh_docs()

    if not st.session_state.documents:
        st.info(
            "📭 No documents yet — and that's OK!\n\n"
            "Go to **💬 Chat** and ask any medical question. "
            "The assistant will answer using web search.",
            icon="💡",
        )
        return

    total = sum(d.get("chunk_count",0) for d in st.session_state.documents)
    c1,c2,c3 = st.columns(3)
    c1.metric("📄 Documents", len(st.session_state.documents))
    c2.metric("🔖 Total Chunks", total)
    c3.metric("🌐 Web Search", "Also active")

    st.markdown("")
    for doc in st.session_state.documents:
        a,b,c,d_ = st.columns([4,2,1,1])
        a.markdown(f"📄 **{doc['filename']}**")
        b.code(doc["doc_id"][:12]+"…", language=None)
        c.markdown(f"🔖 {doc.get('chunk_count','—')}")
        with d_:
            if st.button("🗑️", key=f"del_{doc['doc_id']}",
                         help=f"Delete {doc['filename']}"):
                with st.spinner("Deleting…"):
                    try:
                        st.session_state.client.delete_document(doc["doc_id"])
                        st.session_state.flash = f"🗑️ **{doc['filename']}** deleted."
                        refresh_docs()
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Delete failed: {exc}")


# ── ABOUT ─────────────────────────────────────────────────────────
def show_about_page():
    st.markdown("""
    <div class="rag-header">
        <h2>ℹ️ About</h2>
        <p>Clinical RAG Assistant v2.0 — Web Search + Document RAG · Open-source · Local embeddings</p>
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
### 🔄 How It Works

**Mode 1 — No documents (default)**
```
User question
  ↓
Tavily web search
  ↓ (trusted medical sites)
BeautifulSoup fallback
  ↓
Groq LLM synthesises answer
  ↓
Response + web citations
```

**Mode 2 — Documents uploaded**
```
User question
  ↓
Pinecone semantic search
  ↓ (if low confidence)
Tavily supplement
  ↓
Groq LLM synthesises answer
  ↓
Response + page citations
```

### 🏗️ Tech Stack
| Layer | Tech |
|-------|------|
| Backend | FastAPI 0.111 |
| Frontend | Streamlit |
| LLM | Groq Llama 3 70B |
| Web Search | Tavily API |
| Scraping | BeautifulSoup + lxml |
| Embeddings | BAAI/bge-small-en-v1.5 |
| Vector DB | Pinecone / In-Memory |
| Workflow | LangGraph |
""")
    with col2:
        st.markdown("""
### 🛡️ Safety Guardrails
| Layer | What it checks |
|-------|----------------|
| Input (regex) | Injection, jailbreak, diagnosis |
| Safety Agent | LLM clinical advice detection |
| Retrieval | Score threshold, source validity |
| Output | Prescriptions, dosages in output |

### ✅ Can answer
- Disease mechanisms & pathophysiology
- Anatomy & physiology
- Pharmacology concepts
- Lab value interpretations (educational)
- Medical terminology
- Public health & prevention

### ❌ Will NOT answer
- Diagnose your condition
- Recommend specific medications
- Suggest dosages
- Create treatment plans
- Give emergency advice
""")

    st.warning(
        "⚕️ **Disclaimer:** Educational use only. Not a medical device. "
        "Always consult a qualified healthcare professional.", icon="⚠️")

    st.markdown("---")
    st.markdown("### ⚙️ Live Status")
    try:
        h = st.session_state.client.health()
        cols = st.columns(len(h))
        for col,(k,v) in zip(cols, h.items()):
            col.metric(k.replace("_"," ").title(), str(v))
    except Exception:
        st.error("Backend unreachable.")


# ── MAIN ──────────────────────────────────────────────────────────
def main():
    if not st.session_state.authenticated:
        show_login()
        return
    page = show_sidebar()
    {"Chat": show_chat_page,
     "Documents": show_documents_page,
     "About": show_about_page}.get(page, show_chat_page)()


if __name__ == "__main__":
    main()
