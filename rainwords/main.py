import uvicorn
import sys

# Windows consoles default to cp1252, which cannot encode the ✓/⚠ glyphs this
# module prints at startup — that raises UnicodeEncodeError (previously mistaken
# for a fatal FAISS-load error). Force UTF-8 output.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

print("--- DEBUG: Starting main.py ---", file=sys.stderr)
import faiss
import pickle
import numpy as np
import nltk
import re # Import re
import webbrowser
from fastapi import FastAPI, HTTPException, Header, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import io
import random
import os
from pathlib import Path
from fastapi.responses import FileResponse
from dotenv import load_dotenv
from .cloudflare_embedder import create_embedder
from .text_pipeline import clean_text, normalize_basic, extract_pdf_text, detect_language
from .user_corpora import (
    normalize_handle,
    add_corpus,
    search_owner,
    list_owner_corpora,
    get_owner_docs,
    delete_owner_corpus,
)
from .rarity import is_rare, is_common, rarity_weight, weighted_order

# Load environment variables from .env file if present
# We explicitly look for .env in the same directory as main.py
BASE_DIR = Path(__file__).resolve().parent
env_path = BASE_DIR / ".env"
print(f"DEBUG: Looking for .env at {env_path}")

if env_path.exists():
    print("DEBUG: .env file found.")
    load_dotenv(env_path, override=True)
else:
    print("DEBUG: .env file NOT found. Please ensure it is named exactly '.env' (no .txt extension).")

# Verify key load
key_check = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
if key_check:
    print(f"DEBUG: API Key loaded successfully (Length: {len(key_check)})")
else:
    print("DEBUG: Neither GEMINI_API_KEY nor GOOGLE_API_KEY found in environment variables.")


# Import our helper functions
from .semantics_and_colors import (
    get_colorspace_analysis,
    get_colorspace_analysis_batch,
    extract_keywords,
    MODE_KEYS,
    is_good_word_form,     # NEW
    init_semantics_model   # NEW: Init function
)
from .llm_selection import (
    select_words_with_llm, 
    generate_shadow_poem, 
    trace_roots_with_llm,
    find_amphibians_with_llm
)

# --- Configuration ---
# Base directory of this package (…/site-packages/rainwords)
BASE_DIR = Path(__file__).resolve().parent

INDEX_FILE = BASE_DIR / "poetry.index"
DOCS_FILE  = BASE_DIR / "poetry_docs.pkl"

MODEL_NAME = 'all-MiniLM-L6-v2'

# Word rarity is now computed from general-language frequency (see rarity.py),
# so no corpus-derived word-frequency cache is loaded here.

# --- Application Startup: Load Models ---
# These models are loaded ONCE when the server starts,
# making our API calls very fast.

print("Downloading NLTK data (if needed)...")

# Tokenizer
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Old tagger name (backwards compatibility)
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

# Newer NLTK versions use this name:
try:
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
except LookupError:
    nltk.download('averaged_perceptron_tagger_eng')
    
# NEW: stopwords
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

print("NLTK data is ready.")


print(f"Loading embedding model '{MODEL_NAME}'...")
try:
    EMBEDDING_MODEL = create_embedder(MODEL_NAME)
    model_dim = EMBEDDING_MODEL.get_sentence_embedding_dimension()
    print(f"Embedding model loaded. Dimension: {model_dim}")
    
    # Initialize semantics module with the SAME model instance
    init_semantics_model(EMBEDDING_MODEL)

    # Dimension of the active embedder — user shards must match this to be searchable.
    EMBED_DIM = EMBEDDING_MODEL.get_sentence_embedding_dimension()

except Exception as e:
    print(f"FATAL: Could not load embedding model. Error: {e}")
    exit()

print(f"Loading vector database from '{INDEX_FILE}'...")
try:
    VECTOR_INDEX = faiss.read_index(str(INDEX_FILE))
    index_dim = VECTOR_INDEX.d
    print(f"FAISS index loaded. Dimension: {index_dim}, Total vectors: {VECTOR_INDEX.ntotal}")
    
    # Check dimension match
    if hasattr(EMBEDDING_MODEL, 'get_sentence_embedding_dimension'):
        model_dim = EMBEDDING_MODEL.get_sentence_embedding_dimension()
        if index_dim != model_dim:
            print(f"\n⚠️  WARNING: Dimension mismatch!")
            print(f"   Model dimension: {model_dim}")
            print(f"   Index dimension: {index_dim}")
            print(f"   You need to rebuild the FAISS index!\n")
except Exception as e:
    print(f"FATAL: Could not load FAISS index. Did you run 'build_index.py'? Error: {e}")
    exit()

print(f"Loading document map from '{DOCS_FILE}'...")
try:
    with open(DOCS_FILE, "rb") as f:
        DOCUMENTS = pickle.load(f)
    print(f"Document map loaded. Total documents: {len(DOCUMENTS)}")
    sources = sorted({doc["source"] for doc in DOCUMENTS})
    BUILTIN_SOURCES = {s.lower() for s in sources}
    print("Available corpus sources in DOCUMENTS:")
    for s in sources:
        print("  •", repr(s))
except Exception as e:
    print(f"FATAL: Could not load document map. Did you run 'corpus_builder'? Error: {e}")
    exit()


# ----------------------------------------------------
    

import re

LETTER_CLASS = r"A-Za-zÀ-ÖØ-öø-ÿ"
WORD_FORM_RE = re.compile(
    rf"^[{LETTER_CLASS}][{LETTER_CLASS}'’\-]*[{LETTER_CLASS}]$",
    flags=re.UNICODE,
)





def colorspace_to_vector(cs: dict, mode: str) -> np.ndarray:
    """
    Turn a colorspace dict (e.g., {"fire":0.7, "water":0.3}) into a fixed vector,
    using the canonical keys from semantics_and_colors.MODE_KEYS.
    """
    if cs is None:
        return np.zeros(1, dtype=float)

    norm_mode = mode.lower().strip().replace(" ", "_")
    keys = MODE_KEYS.get(norm_mode)

    if keys is None:
        # fallback: sorted keys from whatever came back
        keys = sorted(cs.keys())

    return np.array([cs.get(k, 0.0) for k in keys], dtype=float)



def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    if v1.size == 0 or v2.size == 0:
        return 0.0
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))


# --- Alchemy: how multiple selected corpora are combined ---

def _balance_by_source(docs: list, sources: set, target_count: int) -> list:
    """
    Weave/Fusion: take a balanced, interleaved set of docs across the selected
    corpora so each is represented (instead of the global top being dominated by
    the closest corpus).
    """
    from collections import OrderedDict
    srcs = sorted(sources)
    k = max(1, len(srcs))
    per = max(3, (target_count + k - 1) // k)   # docs per corpus to draw words from
    groups = OrderedDict((s, []) for s in srcs)
    for d in docs:
        lst = groups.get(d.get("source", "").lower())
        if lst is not None and len(lst) < per:
            lst.append(d)
    lists = [v for v in groups.values() if v]
    out, i = [], 0
    while any(i < len(lst) for lst in lists):
        for lst in lists:
            if i < len(lst):
                out.append(lst[i])
        i += 1
    return out


def _clean_keywords(text, q_lang, pos, user_words, rarity):
    """Ordered, filtered candidate keywords from one stanza (shared by Fusion)."""
    out = []
    for kw in extract_keywords(text, lang=q_lang, pos=pos):
        lw = kw.lower()
        if not is_good_word_form(lw) or lw in user_words:
            continue
        if rarity == "only_rare" and not is_rare(lw, q_lang):
            continue
        if rarity == "prefer_rare" and is_common(lw, q_lang):
            continue
        if rarity == "prefer_common" and is_rare(lw, q_lang):
            continue
        out.append(kw)
    return out


def _per_source_words(docs, q_lang, pos, user_words, rarity):
    """
    Ordered, de-duped candidate words grouped by their source corpus, plus a
    word -> (source, snippet) provenance map. Shared by Weave and Fusion.
    """
    from collections import OrderedDict
    per_source = OrderedDict()   # key(lower) -> {"src", "words", "seen"}
    provenance = {}
    for d in docs:
        src = d.get("source", "")
        entry = per_source.setdefault(src.lower(), {"src": src, "words": [], "seen": set()})
        for kw in _clean_keywords(d["text"], q_lang, pos, user_words, rarity):
            lw = kw.lower()
            if lw not in entry["seen"]:
                entry["seen"].add(lw)
                entry["words"].append(kw)
            if lw not in provenance:
                provenance[lw] = (src, _snippet_for(kw, d["text"]))
    return per_source, provenance


def _weave_order(per_source):
    """Round-robin one word from each corpus in turn — equal parts of each."""
    lists = [e["words"] for e in per_source.values()]
    out, seen, i = [], set(), 0
    while any(i < len(lst) for lst in lists):
        for lst in lists:
            if i < len(lst):
                kw = lst[i]
                lw = kw.lower()
                if lw not in seen:
                    seen.add(lw)
                    out.append(kw)
        i += 1
    return out


def _fusion_order(per_source):
    """Rank words by how many corpora surface them (then by best position)."""
    tally = {}
    for entry in per_source.values():
        for idx, kw in enumerate(entry["words"]):
            lw = kw.lower()
            t = tally.setdefault(lw, {"kw": kw, "n": 0, "pos": 10 ** 9})
            t["n"] += 1
            t["pos"] = min(t["pos"], idx)
    ranked = sorted(tally.values(), key=lambda t: (-t["n"], t["pos"]))
    return [t["kw"] for t in ranked]


def _snippet_for(word: str, text: str, width: int = 90) -> str:
    """A short fragment of `text` around `word` (whole-word), for the source trail."""
    text = " ".join(text.split())
    m = re.search(r"\b" + re.escape(word) + r"\b", text, re.IGNORECASE)
    i = m.start() if m else text.lower().find(word.lower())
    if i < 0:
        return (text[:width] + "…") if len(text) > width else text
    half = max(0, (width - len(word)) // 2)
    start, end = max(0, i - half), min(len(text), i + len(word) + half)
    snip = text[start:end].strip()
    if start > 0:
        snip = "…" + snip
    if end < len(text):
        snip = snip + "…"
    return snip


# --- Initialize FastAPI App ---
app = FastAPI(title="RainWords AI API")

browser_opened = False

@app.on_event("startup")
def open_browser_event():
    global browser_opened
    if browser_opened:
        return
    browser_opened = True

    url = "http://127.0.0.1:8080"
    try:
        webbrowser.open(url)
    except Exception as e:
        print("Could not open browser:", e)



# Add CORS (Cross-Origin Resource Sharing) middleware
# This allows our index.html (on a file:// or different port)
# to talk to our Python server (on http://localhost:8080)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (for local development)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# --- Pydantic Models: Data Validation ---
# This defines what the JSON from the frontend should look like
# --- Pydantic Models: Data Validation ---
class SuggestionRequest(BaseModel):
    text: str
    colorspace: str                    # "elements" | "temperature" | "chakras"
    attention: str                     # "line" | "full_text"
    k: int = 5
    max_words: int = 10
    corpus: str | None = None          # legacy single
    corpora: list[str] | None = None   # NEW: multi
    pos: List[str] | None = None       # POS control
    lens: str = "semantic"             # "semantic" or "colorspace"
    rarity: str | None = 'off'  # NEW: 'prefer_rare', 'prefer_common', 'only_rare'
    alchemy: str | None = "blend"      # multi-corpus: 'blend' | 'weave' | 'fusion'
    llm_mode: str | None = "none"      # "none", "ollama", "huggingface", "gemini"
    llm_model: str | None = None       # e.g. "llama3", "gemini-1.5-flash"



class WordSuggestion(BaseModel):
    word: str
    colors: Dict[str, float]
    source: Optional[str] = None    # corpus the word was drawn from
    snippet: Optional[str] = None   # the fragment it fell from (for the source trail)


class EdgeInfo(BaseModel):
    a: str           # word A
    b: str           # word B
    sim: float       # semantic similarity between A and B
    direction: int   # -1, 0, or 1


class SuggestionsResponse(BaseModel):
    suggestions: List[WordSuggestion]
    mood: Dict[str, float]
    edges: List[EdgeInfo] = []   # 🔹 new


class ShadowPoemRequest(BaseModel):
    words: List[str]
    text_context: Optional[str] = None  # The full poem text for language detection
    llm_mode: str = "gemini"
    llm_model: str | None = None

class ShadowPoemResponse(BaseModel):
    title: str
    body: str

class RootTraceRequest(BaseModel):
    text: str
    llm_mode: str = "gemini"
    llm_model: str | None = None
    depth: str = "deep"  # "deep" or "standard"

class AmphibianRequest(BaseModel):
    roots: List[str]
    text_context: Optional[str] = None  # The full poem text for language detection
    llm_mode: str = "gemini"
    llm_model: Optional[str] = None

# --- API Endpoints ---

@app.post("/api/shadow_poem", response_model=ShadowPoemResponse)
def create_shadow_poem(request: ShadowPoemRequest):
    print(f"Generating Shadow Poem from {len(request.words)} words...")
    
    result = generate_shadow_poem(
        words=request.words,
        context_text=request.text_context,
        mode=request.llm_mode,
        model_name=request.llm_model,
        api_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    )
    
    return ShadowPoemResponse(title=result.get("title", "Untitled"), body=result.get("body", ""))

@app.post("/api/trace_roots")
def trace_roots(request: RootTraceRequest):
    print(f"Tracing roots for text length {len(request.text)} (Depth: {request.depth})...")
    result = trace_roots_with_llm(
        text=request.text,
        depth=request.depth,
        mode=request.llm_mode,
        model_name=request.llm_model,
        api_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    )
    return result

@app.post("/api/find_amphibians")
def find_amphibians(request: AmphibianRequest):
    print(f"Finding amphibians for {len(request.roots)} roots...")
    result = find_amphibians_with_llm(
        roots_list=request.roots,
        context_text=request.text_context,
        mode=request.llm_mode,
        model_name=request.llm_model,
        api_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    )
    return result


@app.get("/")
def serve_frontend():
    index = Path(__file__).parent / "main.html"
    return FileResponse(index)

@app.get("/api/status")
def read_root():
    return {"status": "RainWords API is running."}


@app.post("/api/suggestions", response_model=SuggestionsResponse)
def get_suggestions(
    request: SuggestionRequest,
    x_rainwords_handle: Optional[str] = Header(default=None),
):
    """
    The main endpoint for getting AI-powered word suggestions.
    """
    owner = normalize_handle(x_rainwords_handle)
    print(f"\nReceived suggestion request:")
    print(f"  - Colorspace: {request.colorspace}")
    print(f"  - Attention: {request.attention}")
    print(f"  - Corpus: {repr(request.corpus)}")
    print(f"  - Lens: {request.lens}")
    print(f"  - Rarity: {request.rarity}")
    print(f"  - Handle: {repr(owner) if owner else '(none)'}")


    try:
        # 1. Get query text based on "attention"
        full_text = request.text.strip()
        if not full_text:
            return SuggestionsResponse(suggestions=[], mood={})


        if request.attention == "line":
            # get last non-empty line
            lines = [ln for ln in full_text.split("\n") if ln.strip()]
            if not lines:
                return SuggestionsResponse(suggestions=[], mood={})
            query_text = lines[-1]
        else:
            query_text = full_text

        print(f'  - Querying with text: "{query_text[:50]}..."')

        # Mood of the verse/poem in the chosen colorspace
        verse_mood = get_colorspace_analysis(query_text, request.colorspace)

        

        # 2. Build allowed_sources from corpus filters
        allowed_sources: set[str] | None = None
        if request.corpora:                       # multi
            allowed_sources = {s.lower() for s in request.corpora}
        elif request.corpus:                      # single (legacy)
            allowed_sources = {request.corpus.lower()}

        # We want more candidates than final words
        target_count = max(request.k, request.max_words * 3)

        query_embedding = EMBEDDING_MODEL.encode([query_text]).astype("float32")[0]

        # 3. Retrieve candidates from the built-in index AND this handle's
        #    uploaded shards, ranked together by embedding distance. The source
        #    filter is applied per source BEFORE truncation, so a small selected
        #    corpus is never crowded out of a top-K by larger corpora.
        q_vec = query_embedding.reshape(1, -1)  # FAISS wants (1, d)
        wants_builtin = (not allowed_sources) or bool(allowed_sources & BUILTIN_SOURCES)

        candidates: list[tuple[float, dict]] = []
        if wants_builtin:
            # Cover the whole index when filtering so a small corpus isn't lost.
            search_k = VECTOR_INDEX.ntotal if allowed_sources else min(
                VECTOR_INDEX.ntotal, max(request.k * 10, 80))
            D, I = VECTOR_INDEX.search(q_vec, k=search_k)
            candidates = [
                (float(D[0][rank]), DOCUMENTS[idx])
                for rank, idx in enumerate(I[0]) if idx != -1
            ]
        if owner:
            owner_k = 10 ** 9 if allowed_sources else min(
                VECTOR_INDEX.ntotal, max(request.k * 10, 80))
            candidates += search_owner(
                owner, query_embedding, EMBED_DIM, owner_k, allowed_sources
            )

        if allowed_sources:
            candidates = [
                (dist, doc) for (dist, doc) in candidates
                if doc["source"].lower() in allowed_sources
            ]
        candidates.sort(key=lambda x: x[0])

        alchemy = (request.alchemy or "blend").lower()
        multi_corpus = bool(allowed_sources) and len(allowed_sources) >= 2

        if request.lens == "colorspace":
            # Re-rank the nearest N by colorspace (mood) similarity. Cap the pool
            # so the per-doc mood analysis (an embedding each) stays bounded even
            # when a large corpus is selected. (Could be batched later for speed.)
            pool = [doc for (dist, doc) in candidates[:max(target_count * 2, 60)]]
            cs_q = get_colorspace_analysis(query_text, request.colorspace)
            v_q = colorspace_to_vector(cs_q, request.colorspace)
            scored: list[tuple[float, dict]] = []
            for doc in pool:
                cs_d = get_colorspace_analysis(doc["text"], request.colorspace)
                v_d = colorspace_to_vector(cs_d, request.colorspace)
                scored.append((cosine_similarity(v_q, v_d), doc))
            scored.sort(key=lambda x: x[0], reverse=True)
            ranked_docs = [doc for (sim, doc) in scored]
        else:
            ranked_docs = [doc for (dist, doc) in candidates]

        # Alchemy: Weave/Fusion balance the pool across the selected corpora so a
        # closer corpus can't dominate the result; Blend keeps the global best.
        if alchemy in ("weave", "fusion") and multi_corpus:
            retrieved_docs = _balance_by_source(ranked_docs, allowed_sources, target_count)
        else:
            retrieved_docs = ranked_docs[:target_count]

        # 4. Retrieve & extract keywords
        print("Allowed sources:", allowed_sources if allowed_sources else "(ALL)")
        print("\n--- Retrieved Stanzas (in similarity order) ---")
        for rank, doc in enumerate(retrieved_docs):
            print(f"[{rank+1}] ({doc['source']}): {doc['text']}")
        print("------------------------------------------------\n")

        user_words = set(re.findall(r"\b\w+\b", full_text.lower()))
        q_lang = detect_language(full_text)   # 'fr' | 'en' for wordfreq-based rarity
        final_keywords: list[str] = []
        seen: set[str] = set()
        word_provenance: dict = {}   # word -> (source, snippet) for the source trail
        max_per_stanza = 3

        # NEW: LLM Mode check
        use_llm = request.llm_mode and request.llm_mode.lower() != "none"
        
        # If using LLM, we want to collect many candidates first
        llm_candidates: list[str] = []
        llm_candidate_limit = request.max_words * 5

        # Weave / Fusion alchemy select words from per-corpus lists (Weave =
        # round-robin one per corpus; Fusion = words shared across corpora),
        # instead of the per-stanza scan below — so Weave is truly balanced.
        alchemy_active = alchemy in ("weave", "fusion") and multi_corpus
        if alchemy_active:
            rarity = (request.rarity or "off").lower()
            per_source, alch_prov = _per_source_words(
                retrieved_docs, q_lang, request.pos, user_words, rarity)
            word_provenance.update(alch_prov)
            ordered = (_fusion_order(per_source) if alchemy == "fusion"
                       else _weave_order(per_source))
            if use_llm:
                llm_candidates = ordered[:llm_candidate_limit]
            else:
                for kw in ordered:
                    lw = kw.lower()
                    if lw in seen:
                        continue
                    seen.add(lw)
                    final_keywords.append(kw)
                    if len(final_keywords) >= request.max_words:
                        break

        for doc in retrieved_docs:
            if alchemy_active:
                break   # Weave/Fusion already produced the words above
            # Stop condition for Random mode
            if not use_llm and len(final_keywords) >= request.max_words:
                break
            # Stop condition for LLM mode (collecting candidates)
            if use_llm and len(llm_candidates) >= llm_candidate_limit:
                break

            stanza_text = doc["text"]
            stanza_keywords = list(
                extract_keywords(stanza_text, lang=q_lang, pos=request.pos)
            )

            rarity = (request.rarity or "off").lower()

            stanza_clean: list[str] = []
            for kw in stanza_keywords:
                lw = kw.lower()

                # 🔹 1) drop ugly tokens first
                if not is_good_word_form(lw):
                    continue

                # 🔹 2) don’t repeat user words or already-used words
                if lw in user_words:
                    continue
                
                # For Random mode, we check 'seen' here.
                # For LLM mode, we also check 'seen' to avoid duplicates in candidate list
                if lw in seen:
                    continue

                # 🔹 3) rarity filtering (general-language rarity via wordfreq,
                #    so it behaves the same on built-in and uploaded corpora).
                #    only_rare -> keep only rare words (hard).
                #    prefer_*  -> drop the clearly-opposite band (hard), then bias
                #                 softly toward the preference during selection.
                if rarity == "only_rare":
                    if not is_rare(lw, q_lang):
                        continue
                elif rarity == "prefer_rare":
                    if is_common(lw, q_lang):
                        continue
                elif rarity == "prefer_common":
                    if is_rare(lw, q_lang):
                        continue

                stanza_clean.append(kw)
                if lw not in word_provenance:
                    word_provenance[lw] = (doc["source"], _snippet_for(kw, stanza_text))

            # --- Selection Logic ---
            if use_llm:
                # In LLM mode, we just add valid words to candidates
                # We preserve stanza priority by appending in order
                for kw in stanza_clean:
                    if kw.lower() not in seen:
                        llm_candidates.append(kw)
                        seen.add(kw.lower())
            else:
                # Random mode: soft-bias the order by rarity when a preference
                # is set, otherwise a plain shuffle.
                if rarity in ("prefer_rare", "prefer_common"):
                    weights = [rarity_weight(kw, q_lang, rarity) for kw in stanza_clean]
                    ordered = weighted_order(stanza_clean, weights)
                else:
                    ordered = list(stanza_clean)
                    random.shuffle(ordered)
                for kw in ordered[:max_per_stanza]:
                    lw = kw.lower()
                    if lw in seen:
                        continue
                    seen.add(lw)
                    final_keywords.append(kw)
                    if len(final_keywords) >= request.max_words:
                        break
        
        # If LLM mode, now perform the selection
        if use_llm:
            rarity = (request.rarity or "off").lower()
            if rarity in ("prefer_rare", "prefer_common"):
                # Present candidates rarest/commonest first so the LLM prioritizes.
                llm_candidates.sort(
                    key=lambda w: rarity_weight(w, q_lang, rarity), reverse=True
                )
            print(f"  - LLM Selection Mode: {request.llm_mode}")
            print(f"  - Candidates ({len(llm_candidates)}): {llm_candidates}")
            
            final_keywords = select_words_with_llm(
                candidates=llm_candidates,
                count=request.max_words,
                query_text=query_text,
                mode=request.llm_mode,
                model_name=request.llm_model,
                api_key=os.environ.get("GEMINI_API_KEY")
            )
            print(f"  - LLM Selected: {final_keywords}")

        print(f"  - Selected {len(final_keywords)} keywords: {final_keywords}")

        # 5. Colors + provenance for each keyword (BATCH OPTIMIZED)
        final_suggestions: list[WordSuggestion] = []

        def _prov(w):
            p = word_provenance.get(w.lower())
            return {"source": p[0], "snippet": p[1]} if p else {}

        try:
            # Batch analyze colors
            color_analyses = get_colorspace_analysis_batch(final_keywords, request.colorspace)

            for word, color_data in zip(final_keywords, color_analyses):
                final_suggestions.append(
                    WordSuggestion(word=word, colors=color_data, **_prov(word))
                )
        except Exception as e:
            print(f"    - Error in batch color analysis: {e}")
            # Fallback to individual (though batch handles errors internally usually)
            for word in final_keywords:
                try:
                    color_data = get_colorspace_analysis(word, request.colorspace)
                    final_suggestions.append(WordSuggestion(word=word, colors=color_data, **_prov(word)))
                except Exception:
                    final_suggestions.append(WordSuggestion(word=word, colors={"air": 1.0}, **_prov(word)))

        # 6. Build semantic edges between suggestion words
        edges: list[EdgeInfo] = []

        try:
            if final_keywords:
                # query embedding is already computed as `query_embedding`
                q_vec = query_embedding  # shape (d,)

                # Batch encode word vectors for edge calculation
                word_vecs = EMBEDDING_MODEL.encode(final_keywords).astype("float32")
                
                # Compute similarities to query
                # q_vec is (d,), word_vecs is (N, d)
                # We can use dot product if normalized, or cosine_similarity manually
                # sentence_transformers.util.cos_sim is convenient
                from sentence_transformers import util
                sims_to_query = util.cos_sim(word_vecs, q_vec).flatten().tolist()

                n = len(final_keywords)
                for i in range(n):
                    for j in range(i + 1, n):
                        # sim_ij = cosine_similarity(word_vecs[i], word_vecs[j])
                        # Use dot product since vectors are normalized by ST usually? 
                        # Actually let's stick to util.cos_sim for safety
                        sim_ij = util.cos_sim(word_vecs[i], word_vecs[j]).item()
                        
                        if sim_ij <= 0:
                            continue  # skip negative or zero similarity if you want

                        s_i = sims_to_query[i]
                        s_j = sims_to_query[j]
                        delta = s_j - s_i

                        # small difference -> treat as undirected
                        if abs(delta) < 0.02:
                            direction = 0
                        else:
                            # 1 means flow i -> j, -1 means j -> i
                            direction = 1 if delta > 0 else -1

                        edges.append(
                            EdgeInfo(
                                a=final_keywords[i],
                                b=final_keywords[j],
                                sim=float(sim_ij),
                                direction=direction,
                            )
                        )
        except Exception as e:
            print("Could not compute edges:", e)
            edges = []

        print(f"  - Returning {len(final_suggestions)} suggestions to frontend.")
        return SuggestionsResponse(
            suggestions=final_suggestions,
            mood=verse_mood,
            edges=edges,
        )



    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"\n=== ERROR in /api/suggestions ===")
        print(f"Error: {e}")
        print(f"Traceback:\n{error_trace}")
        print(f"==================================\n")
        raise HTTPException(status_code=500, detail=str(e))



CORPUS_DIR = "corpuses"  # or whatever you use now

# Uploads: accept .pdf / .txt up to this size (no sign-in => be defensive).
MAX_UPLOAD_BYTES = 20 * 1024 * 1024


@app.get("/api/corpora")
def list_corpora(x_rainwords_handle: Optional[str] = Header(default=None)):
    """
    Return the built-in corpus source labels, plus any corpora uploaded under
    the caller's handle. Other handles' uploads are never listed.
    Example: {"corpora": [...all...], "builtin": [...], "mine": [...]}
    """
    try:
        builtin = sorted({doc["source"] for doc in DOCUMENTS})
    except Exception:
        builtin = []

    owner = normalize_handle(x_rainwords_handle)
    mine: list[str] = []
    if owner:
        try:
            mine = list_owner_corpora(owner, EMBED_DIM)
        except Exception as e:
            print(f"Could not list corpora for handle {owner!r}: {e}")

    combined = builtin + [m for m in mine if m not in builtin]
    return {"corpora": combined, "builtin": builtin, "mine": mine}


@app.post("/api/corpora/upload")
async def upload_corpus(
    file: UploadFile = File(...),
    handle: str = Form(...),
):
    """
    Upload a .pdf or .txt corpus under a claimed handle. The file is sanitized,
    chunked, embedded (with the same model as the built-in index), and persisted
    as a per-owner shard so it survives restarts and reappears on reconnect.
    """
    owner = normalize_handle(handle)
    if not owner:
        raise HTTPException(status_code=400, detail="A valid handle is required.")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file.")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large (max {MAX_UPLOAD_BYTES // (1024 * 1024)} MB).",
        )

    name = file.filename or "corpus"
    lower = name.lower()

    if lower.endswith(".pdf"):
        try:
            extracted = extract_pdf_text(io.BytesIO(raw))
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Could not read PDF: {e}")
        ready_text = clean_text(extracted)
    elif lower.endswith(".txt"):
        # Preserve the blank-line stanza structure; just normalize glyphs.
        ready_text = normalize_basic(raw.decode("utf-8", errors="replace"))
    else:
        raise HTTPException(
            status_code=400, detail="Only .pdf and .txt files are supported."
        )

    if not ready_text.strip():
        raise HTTPException(
            status_code=422,
            detail="No text could be extracted. If this is a scanned PDF, it has no selectable text (OCR is not supported).",
        )

    # Label the corpus with its detected language so the UI groups it under
    # FR/EN correctly (matches the built-in "(FR)"/"(EN)" naming convention).
    lang = detect_language(ready_text).upper()   # "FR" or "EN"
    label = f"{Path(name).stem} ({lang}).txt"

    try:
        meta = add_corpus(owner, label, ready_text, EMBEDDING_MODEL)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")

    return {
        "ok": True,
        "corpus": meta["label"],
        "n_chunks": meta["n_chunks"],
        "corpus_id": meta["corpus_id"],
    }


@app.delete("/api/corpora")
def delete_corpus(
    label: str,
    x_rainwords_handle: Optional[str] = Header(default=None),
):
    """Delete one of the caller's uploaded corpora (by its source label)."""
    owner = normalize_handle(x_rainwords_handle)
    if not owner:
        raise HTTPException(status_code=400, detail="A valid handle is required.")
    if delete_owner_corpus(owner, label):
        return {"ok": True, "deleted": label}
    raise HTTPException(status_code=404, detail="Corpus not found for this handle.")



# --- Run the Server ---
if __name__ == "__main__":
    print("Starting Uvicorn server on http://127.0.0.1:8080")
    uvicorn.run(app, host="127.0.0.1", port=8080)