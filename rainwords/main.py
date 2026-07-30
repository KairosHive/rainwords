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

# NEW: cache file for word frequencies
WORD_FREQ_FILE = BASE_DIR / "word_freq.pkl"

try:
    with open(WORD_FREQ_FILE, "rb") as f:
        wf_payload = pickle.load(f)
        WORD_FREQ = wf_payload.get("freq", {})
        RARE_CUT = float(wf_payload.get("rare_cut", 1.0))
        COMMON_CUT = float(wf_payload.get("common_cut", 4.0))

    print(
        f"Loaded word frequency cache: {len(WORD_FREQ)} words, "
        f"rare_cut={RARE_CUT}, common_cut={COMMON_CUT}"
    )
except Exception as e:
    print(f"Warning: could not load word frequency cache: {e}")
    WORD_FREQ = {}
    RARE_CUT = 1.0
    COMMON_CUT = 4.0

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
    llm_mode: str | None = "none"      # "none", "ollama", "huggingface", "gemini"
    llm_model: str | None = None       # e.g. "llama3", "gemini-1.5-flash"



class WordSuggestion(BaseModel):
    word: str
    colors: Dict[str, float]


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
        #    uploaded shards, then merge. Both paths yield (distance, doc)
        #    tuples (squared-L2) so they can be ranked together.
        q_vec = query_embedding.reshape(1, -1)  # FAISS wants (1, d)

        if request.lens == "colorspace":
            # Big candidate pool, then re-rank by colorspace vector similarity.
            base_k = max(target_count * 10, 100) if allowed_sources else max(target_count * 5, 60)
            search_k = min(VECTOR_INDEX.ntotal, base_k)
            D, I = VECTOR_INDEX.search(q_vec, k=search_k)

            cand_docs = [DOCUMENTS[idx] for idx in I[0] if idx != -1]
            if owner:
                cand_docs += get_owner_docs(owner, EMBED_DIM)

            # apply corpus filter (if any)
            if allowed_sources:
                cand_docs = [d for d in cand_docs if d["source"].lower() in allowed_sources]

            # colorspace re-ranking
            cs_q = get_colorspace_analysis(query_text, request.colorspace)
            v_q = colorspace_to_vector(cs_q, request.colorspace)

            scored: list[tuple[float, dict]] = []
            for doc in cand_docs:
                cs_d = get_colorspace_analysis(doc["text"], request.colorspace)
                v_d = colorspace_to_vector(cs_d, request.colorspace)
                sim = cosine_similarity(v_q, v_d)
                scored.append((sim, doc))

            scored.sort(key=lambda x: x[0], reverse=True)
            retrieved_docs = [doc for (sim, doc) in scored[:target_count]]

        else:
            # Semantic lens: nearest neighbours by embedding distance.
            if allowed_sources:
                search_k = min(VECTOR_INDEX.ntotal, max(request.k * 20, 100))
            else:
                search_k = min(VECTOR_INDEX.ntotal, max(request.k * 10, 80))

            D, I = VECTOR_INDEX.search(q_vec, k=search_k)
            candidates: list[tuple[float, dict]] = [
                (float(D[0][rank]), DOCUMENTS[idx])
                for rank, idx in enumerate(I[0]) if idx != -1
            ]
            if owner:
                candidates += search_owner(owner, query_embedding, EMBED_DIM, search_k)

            if allowed_sources:
                candidates = [
                    (dist, doc) for (dist, doc) in candidates
                    if doc["source"].lower() in allowed_sources
                ]

            candidates.sort(key=lambda x: x[0])
            retrieved_docs = [doc for (dist, doc) in candidates[:target_count]]

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
        max_per_stanza = 3

        # NEW: LLM Mode check
        use_llm = request.llm_mode and request.llm_mode.lower() != "none"
        
        # If using LLM, we want to collect many candidates first
        llm_candidates: list[str] = []
        llm_candidate_limit = request.max_words * 5

        for doc in retrieved_docs:
            # Stop condition for Random mode
            if not use_llm and len(final_keywords) >= request.max_words:
                break
            # Stop condition for LLM mode (collecting candidates)
            if use_llm and len(llm_candidates) >= llm_candidate_limit:
                break

            stanza_text = doc["text"]
            stanza_keywords = list(
                extract_keywords(stanza_text, lang=None, pos=request.pos)
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

        # 5. Colors for each keyword (BATCH OPTIMIZED)
        final_suggestions: list[WordSuggestion] = []
        
        try:
            # Batch analyze colors
            color_analyses = get_colorspace_analysis_batch(final_keywords, request.colorspace)
            
            for word, color_data in zip(final_keywords, color_analyses):
                final_suggestions.append(
                    WordSuggestion(word=word, colors=color_data)
                )
        except Exception as e:
            print(f"    - Error in batch color analysis: {e}")
            # Fallback to individual (though batch handles errors internally usually)
            for word in final_keywords:
                try:
                    color_data = get_colorspace_analysis(word, request.colorspace)
                    final_suggestions.append(WordSuggestion(word=word, colors=color_data))
                except Exception:
                    final_suggestions.append(WordSuggestion(word=word, colors={"air": 1.0}))

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



# --- Run the Server ---
if __name__ == "__main__":
    print("Starting Uvicorn server on http://127.0.0.1:8080")
    uvicorn.run(app, host="127.0.0.1", port=8080)