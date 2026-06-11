"""
Web search module: Tavily API + BeautifulSoup scraping fallback.

Guardrail layers applied to web content:
  1. Query guardrail   — block unsafe queries BEFORE calling Tavily
  2. Domain whitelist  — only trusted medical education domains allowed
  3. Content filter    — strip prescription/dosage content from raw results
  4. Source scorer     — penalise low-trust domains, boost .gov/.edu
  5. Result guardrail  — drop results that contain unsafe clinical content
"""
import re
from typing import Optional
from loguru import logger

try:
    from tavily import TavilyClient
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False
    logger.warning("tavily-python not installed")

try:
    import httpx
    from bs4 import BeautifulSoup
    SCRAPING_AVAILABLE = True
except ImportError:
    SCRAPING_AVAILABLE = False

from backend.config import get_settings

settings = get_settings()

# ── Trusted domains (whitelist) ───────────────────────────────────
# Only results from these domains pass domain-level guardrails.
TRUSTED_DOMAINS = {
    # Government / authoritative
    "medlineplus.gov":       1.0,
    "nih.gov":               1.0,
    "cdc.gov":               1.0,
    "who.int":               1.0,
    "nhs.uk":                1.0,
    "pubmed.ncbi.nlm.nih.gov": 1.0,
    # Academic medical centres
    "mayoclinic.org":        0.95,
    "clevelandclinic.org":   0.95,
    "hopkinsmedicine.org":   0.95,
    "ucsf.edu":              0.90,
    # Patient education
    "healthline.com":        0.85,
    "patient.info":          0.85,
    "webmd.com":             0.80,
    "rxlist.com":            0.75,
    "drugs.com":             0.70,
}

# ── Patterns that indicate unsafe clinical content ────────────────
# These patterns in a web result will cause it to be filtered out or
# its unsafe sentences to be removed before it reaches the LLM.
UNSAFE_CONTENT_PATTERNS = [
    # Specific dosage instructions
    r"\b(take|administer|give|inject)\s+\d+\s*(mg|mcg|ml|units?|tablets?|capsules?|drops?)\b",
    r"\b\d+\s*(mg|mcg|g|ml|units?)\s+(twice|once|three times|four times|every \d+ hours?)\b",
    r"\bmaximum\s+(daily\s+)?dose\s+(is|of)\s+\d+\s*(mg|g)\b",
    # Prescription directives
    r"\b(prescribe|prescription for|I recommend (taking|using)|start (the patient on))\b",
    r"\b(the patient should (take|receive|be given))\b",
    # Emergency directives
    r"\b(call 911|go to the ER|seek emergency|call an ambulance)\s+immediately\b",
    # Direct diagnosis statements
    r"\b(you (have|likely have|probably have)|this (is|sounds like|looks like) (cancer|diabetes|heart))\b",
]

# Unsafe content compiled regex
_UNSAFE_RE = re.compile(
    "|".join(UNSAFE_CONTENT_PATTERNS), re.IGNORECASE
)

# Sentences containing these patterns are stripped from web content
_SENTENCE_UNSAFE_RE = re.compile(
    r"[^.!?]*(?:" + "|".join(UNSAFE_CONTENT_PATTERNS) + r")[^.!?]*[.!?]",
    re.IGNORECASE,
)


# ── Guardrail helpers ─────────────────────────────────────────────

def is_trusted_domain(url: str) -> tuple[bool, float]:
    """Return (is_trusted, trust_score) for a URL."""
    url_lower = url.lower()
    for domain, score in TRUSTED_DOMAINS.items():
        if domain in url_lower:
            return True, score
    return False, 0.0


def sanitise_web_content(text: str) -> str:
    """
    Remove sentences containing unsafe clinical content
    (specific dosages, prescriptions, direct emergency directives).
    Returns cleaned text. If > 60% is removed, returns empty string.
    """
    if not text:
        return ""

    original_len = len(text)

    # Split into sentences and filter
    sentences = re.split(r'(?<=[.!?])\s+', text)
    safe_sentences = []
    removed = 0

    for sentence in sentences:
        if _UNSAFE_RE.search(sentence):
            removed += 1
            logger.debug(f"[WebGuardrail] Removed unsafe sentence: {sentence[:80]!r}")
        else:
            safe_sentences.append(sentence)

    cleaned = " ".join(safe_sentences).strip()

    # If more than 60% was removed, the whole result is too risky
    if original_len > 0 and len(cleaned) / original_len < 0.4:
        logger.warning("[WebGuardrail] Result dropped — too much unsafe content")
        return ""

    return cleaned


def check_query_for_web_search(query: str) -> tuple[bool, str]:
    """
    Extra guardrail specifically for web search queries.
    Returns (is_safe, reason).
    Catches edge cases that regex input guardrail might miss when
    the query is phrased as a general question but clearly seeks
    clinical advice.
    """
    q = query.lower().strip()

    clinical_patterns = [
        (r"\bwhat (dose|dosage|mg|milligrams?)\b",
         "Dosage information request"),
        (r"\bhow (much|many) (should i|do i|to) (take|use|give)\b",
         "Dosage instruction request"),
        (r"\bcan i (take|mix|combine|use) .{0,40}(with|and) .{0,30}(mg|pill|tablet|medicine)\b",
         "Drug interaction/dosage request"),
        (r"\b(best|strongest|most effective) (antibiotic|painkiller|medication|drug) for\b",
         "Specific medication recommendation request"),
        (r"\bis .{0,30}(safe|dangerous|okay) (to take|for me|during pregnancy)\b",
         "Personal medication safety request"),
        (r"\b(my|the patient'?s?) (symptoms?|condition|diagnosis|test results?)\b",
         "Patient-specific clinical question"),
    ]

    for pattern, reason in clinical_patterns:
        if re.search(pattern, q, re.IGNORECASE):
            logger.warning(f"[WebGuardrail] Query blocked pre-search: {reason}")
            return False, reason

    return True, ""


# ── WebSearchResult ───────────────────────────────────────────────

class WebSearchResult:
    def __init__(self, title: str, url: str, content: str,
                 score: float = 0.8, trust_score: float = 1.0):
        self.title       = title
        self.url         = url
        self.content     = content          # already sanitised
        self.score       = score
        self.trust_score = trust_score

    @property
    def combined_score(self) -> float:
        return round(self.score * self.trust_score, 3)

    def to_dict(self) -> dict:
        return {
            "chunk_id":      f"web-{abs(hash(self.url)) % 100000:05d}",
            "document_name": self.title,
            "page_number":   0,
            "score":         self.combined_score,
            "excerpt":       self.content[:220] + ("…" if len(self.content) > 220 else ""),
            "url":           self.url,
            "source_type":   "web",
        }


# ── Tavily search with guardrails ─────────────────────────────────

def search_tavily(query: str, max_results: int = 5) -> list[WebSearchResult]:
    """
    Search Tavily with full guardrail pipeline:
      query check → search → domain filter → content sanitise → score
    """
    if not TAVILY_AVAILABLE or not settings.TAVILY_API_KEY:
        logger.warning("Tavily unavailable — skipping")
        return []

    # ── Guardrail 1: check query before sending to Tavily ──────────
    safe, reason = check_query_for_web_search(query)
    if not safe:
        logger.warning(f"[WebGuardrail] Tavily search blocked: {reason}")
        return []

    try:
        client = TavilyClient(api_key=settings.TAVILY_API_KEY)

        # Restrict Tavily to trusted domains
        trusted_list = list(TRUSTED_DOMAINS.keys())

        resp = client.search(
            query=f"{query} medical education",
            search_depth="advanced",
            max_results=max_results + 3,   # fetch extra, we'll filter some
            include_answer=False,
            include_raw_content=False,
            include_domains=trusted_list,  # ← Tavily domain restriction
        )

        results = []
        for r in resp.get("results", []):
            url     = r.get("url", "")
            title   = r.get("title", "Web Result")
            content = r.get("content", "")

            # ── Guardrail 2: domain whitelist ──────────────────────
            trusted, trust_score = is_trusted_domain(url)
            if not trusted:
                logger.debug(f"[WebGuardrail] Dropped untrusted domain: {url}")
                continue

            # ── Guardrail 3: content sanitisation ─────────────────
            clean_content = sanitise_web_content(content)
            if not clean_content:
                logger.debug(f"[WebGuardrail] Dropped result — unsafe content: {url}")
                continue

            results.append(WebSearchResult(
                title=title, url=url,
                content=clean_content,
                score=r.get("score", 0.7),
                trust_score=trust_score,
            ))

        # ── Guardrail 4: sort by combined (relevance × trust) score ─
        results.sort(key=lambda x: x.combined_score, reverse=True)
        results = results[:max_results]

        logger.info(f"[Tavily] {len(results)} safe results for: {query[:60]!r}")
        return results

    except Exception as e:
        logger.error(f"Tavily error: {e}")
        return []


# ── BeautifulSoup scraping fallback with guardrails ───────────────

def scrape_medical_content(query: str, max_results: int = 3) -> list[WebSearchResult]:
    """Scrape trusted medical sites with same guardrail pipeline."""
    if not SCRAPING_AVAILABLE:
        return []

    # ── Guardrail 1: check query ───────────────────────────────────
    safe, reason = check_query_for_web_search(query)
    if not safe:
        logger.warning(f"[WebGuardrail] Scraping blocked: {reason}")
        return []

    site_filter = " OR ".join(f"site:{d}" for d in list(TRUSTED_DOMAINS.keys())[:5])
    search_url  = (f"https://html.duckduckgo.com/html/"
                   f"?q={requests_encode(query)}+{site_filter}")

    results = []
    try:
        headers = {"User-Agent": "Mozilla/5.0 (educational-rag-bot/1.0)"}
        with httpx.Client(timeout=12, follow_redirects=True) as client:
            resp = client.get(search_url, headers=headers)
            if resp.status_code != 200:
                return []
            soup  = BeautifulSoup(resp.text, "lxml")
            links = soup.select("a.result__a")[:max_results + 3]

            for link in links:
                href  = link.get("href", "")
                title = link.get_text(strip=True)

                # ── Guardrail 2: domain whitelist ──────────────────
                trusted, trust_score = is_trusted_domain(href)
                if not trusted:
                    continue

                content = _scrape_page_text(href, client, headers)

                # ── Guardrail 3: content sanitisation ─────────────
                clean = sanitise_web_content(content)
                if not clean:
                    continue

                results.append(WebSearchResult(
                    title=title, url=href,
                    content=clean,
                    score=0.65,
                    trust_score=trust_score,
                ))
                if len(results) >= max_results:
                    break

        results.sort(key=lambda x: x.combined_score, reverse=True)
        logger.info(f"[Scraping] {len(results)} safe results")
    except Exception as e:
        logger.warning(f"Scraping error: {e}")

    return results


def _scrape_page_text(url: str, client, headers: dict, max_chars: int = 2000) -> str:
    try:
        resp = client.get(url, headers=headers, timeout=8)
        if resp.status_code != 200:
            return ""
        soup = BeautifulSoup(resp.text, "lxml")
        for tag in soup(["script","style","nav","footer","header",
                         "aside","noscript","form","button"]):
            tag.decompose()
        main = (soup.find("main") or soup.find("article") or
                soup.find(id=re.compile(r"content|main|article", re.I)) or
                soup.find("body"))
        if not main:
            return ""
        text = re.sub(r"\s+", " ", main.get_text(separator=" ", strip=True))
        return text[:max_chars]
    except Exception:
        return ""


def requests_encode(text: str) -> str:
    """Simple URL-safe encoding without importing urllib."""
    return text.replace(" ", "+").replace("&", "%26")


# ── Wikipedia API fallback ────────────────────────────────────────

def scrape_wikipedia(query: str, max_results: int = 3) -> list[WebSearchResult]:
    """Fallback to Wikipedia API if Tavily and DDG scraping fail."""
    safe, reason = check_query_for_web_search(query)
    if not safe:
        logger.warning(f"[WebGuardrail] Wikipedia scraping blocked: {reason}")
        return []

    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "prop": "extracts",
        "exchars": 1500,
        "exintro": "1",
        "explaintext": "1",
        "generator": "search",
        "gsrsearch": query,
        "gsrlimit": max_results,
        "format": "json"
    }
    headers = {"User-Agent": "ClinicalRAGAssistant/1.0 (bot@example.com)"}
    
    results = []
    try:
        import httpx
        with httpx.Client(timeout=10) as client:
            resp = client.get(url, params=params, headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                pages = data.get("query", {}).get("pages", {})
                for page_id, page_data in pages.items():
                    title = page_data.get("title", "")
                    content = page_data.get("extract", "")
                    
                    clean = sanitise_web_content(content)
                    if not clean:
                        continue
                    
                    results.append(WebSearchResult(
                        title=title, 
                        url=f"https://en.wikipedia.org/wiki/{requests_encode(title)}",
                        content=clean,
                        score=0.7,
                        trust_score=0.85 # Patient education tier
                    ))
        results.sort(key=lambda x: x.combined_score, reverse=True)
        logger.info(f"[Wikipedia] {len(results)} safe results")
    except Exception as e:
        logger.warning(f"Wikipedia API error: {e}")
        
    return results

# ── Main entry point ──────────────────────────────────────────────

def web_search_medical(query: str, max_results: int = 5) -> list[WebSearchResult]:
    """
    Search for medical education info with full guardrail pipeline.
    Guardrails applied:
      ① Query check (pre-search) — block clinical queries before Tavily call
      ② Domain whitelist         — only trusted .gov/.edu/medical sites
      ③ Content sanitisation     — strip dosage/prescription sentences
      ④ Score × trust ranking    — authoritative sources ranked higher
    Falls back to BeautifulSoup scraping if Tavily unavailable.
    """
    results = search_tavily(query, max_results=max_results)

    if not results:
        logger.info("Falling back to scraping (DuckDuckGo)")
        results = scrape_medical_content(query, max_results=3)
        
    if not results:
        logger.info("Falling back to Wikipedia API")
        results = scrape_wikipedia(query, max_results=3)

    if not results:
        logger.warning(f"No safe web results for: {query[:60]!r}")

    return results


def format_web_context(results: list[WebSearchResult]) -> str:
    """Format safe web results as LLM context string."""
    if not results:
        return ""
    parts = []
    for i, r in enumerate(results, 1):
        parts.append(
            f"[Web Source {i}: {r.title} | Trust: {r.trust_score:.0%}]\n"
            f"URL: {r.url}\n"
            f"{r.content}"
        )
    return "\n\n---\n\n".join(parts)
