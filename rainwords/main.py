import uvicorn
import sys

print("--- DEBUG: Starting main.py ---", file=sys.stderr)
import faiss
import pickle
import numpy as np
import nltk
import re # Import re
import webbrowser
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import random
import os
from pathlib import Path
from fastapi.responses import FileResponse
from dotenv import load_dotenv

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
    EMBEDDING_MODEL = SentenceTransformer(MODEL_NAME)
    print("Embedding model loaded.")
    
    # Initialize semantics module with the SAME model instance
    init_semantics_model(EMBEDDING_MODEL)
    
except Exception as e:
    print(f"FATAL: Could not load embedding model. Error: {e}")
    exit()

print(f"Loading vector database from '{INDEX_FILE}'...")
try:
    print(f"Loading vector database from '{INDEX_FILE}'...")
    VECTOR_INDEX = faiss.read_index(str(INDEX_FILE))
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

    url = "http://127.0.0.1:8000"
    try:
        webbrowser.open(url)
    except Exception as e:
        print("Could not open browser:", e)



# Add CORS (Cross-Origin Resource Sharing) middleware
# This allows our index.html (on a file:// or different port)
# to talk to our Python server (on http://localhost:8000)
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
def get_suggestions(request: SuggestionRequest):
    """
    The main endpoint for getting AI-powered word suggestions.
    """
    print(f"\nReceived suggestion request:")
    print(f"  - Colorspace: {request.colorspace}")
    print(f"  - Attention: {request.attention}")
    print(f"  - Corpus: {repr(request.corpus)}")
    print(f"  - Lens: {request.lens}")
    print(f"  - Rarity: {request.rarity}")


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

        # 3. Branch: colorspace lens vs semantic lens
        if request.lens == "colorspace":
            # big FAISS search to get a rich candidate pool
            q_vec = query_embedding.reshape(1, -1)  # FAISS wants (1, d)
            if allowed_sources:
                base_k = max(target_count * 10, 100)
            else:
                base_k = max(target_count * 5, 60)

            search_k = min(VECTOR_INDEX.ntotal, base_k)
            D, I = VECTOR_INDEX.search(q_vec, k=search_k)
            all_indices = list(I[0])

            # apply corpus filter (if any)
            if allowed_sources:
                candidate_indices = [
                    idx for idx in all_indices
                    if DOCUMENTS[idx]["source"].lower() in allowed_sources
                ]
            else:
                candidate_indices = all_indices

            # colorspace re-ranking
            cs_q = get_colorspace_analysis(query_text, request.colorspace)
            v_q = colorspace_to_vector(cs_q, request.colorspace)

            scored: list[tuple[float, int]] = []
            for idx in candidate_indices:
                stanza_text = DOCUMENTS[idx]["text"]
                cs_d = get_colorspace_analysis(stanza_text, request.colorspace)
                v_d = colorspace_to_vector(cs_d, request.colorspace)
                sim = cosine_similarity(v_q, v_d)
                scored.append((sim, idx))

            scored.sort(key=lambda x: x[0], reverse=True)
            retrieved_indices = [idx for (sim, idx) in scored[:target_count]]

        else:
            q_vec = query_embedding.reshape(1, -1)
            if allowed_sources:
                search_k = min(VECTOR_INDEX.ntotal, max(request.k * 20, 100))
            else:
                search_k = min(VECTOR_INDEX.ntotal, max(request.k * 10, 80))

            D, I = VECTOR_INDEX.search(q_vec, k=search_k)
            all_indices = list(I[0])

            if allowed_sources:
                filtered = [
                    idx for idx in all_indices
                    if DOCUMENTS[idx]["source"].lower() in allowed_sources
                ]
                retrieved_indices = filtered[:target_count]
            else:
                retrieved_indices = all_indices[:target_count]

        # from here on, your existing code that uses retrieved_indices continues...
        print("Allowed sources:", allowed_sources if allowed_sources else "(ALL)")
        print(
            "Sample of DOCUMENTS sources:",
            sorted({doc["source"] for doc in DOCUMENTS})[:8]
        )

        # 4. Retrieve & extract keywords (unchanged)
        print("\n--- Retrieved Stanzas (in similarity order) ---")
        for rank, idx in enumerate(retrieved_indices):
            stanza_text = DOCUMENTS[idx]["text"]
            print(f"[{rank+1}] (id={idx}): {stanza_text}")
        print("------------------------------------------------\n")

        user_words = set(re.findall(r"\b\w+\b", full_text.lower()))
        final_keywords: list[str] = []
        seen: set[str] = set()
        max_per_stanza = 3

        # NEW: LLM Mode check
        use_llm = request.llm_mode and request.llm_mode.lower() != "none"
        
        # If using LLM, we want to collect many candidates first
        llm_candidates: list[str] = []
        llm_candidate_limit = request.max_words * 5

        for idx in retrieved_indices:
            # Stop condition for Random mode
            if not use_llm and len(final_keywords) >= request.max_words:
                break
            # Stop condition for LLM mode (collecting candidates)
            if use_llm and len(llm_candidates) >= llm_candidate_limit:
                break

            stanza_text = DOCUMENTS[idx]["text"]
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

                # 🔹 3) rarity filtering
                freq = WORD_FREQ.get(lw, 1)

                if rarity == "off":
                    pass  # no rarity filter
                elif rarity == "only_rare":
                    if freq > RARE_CUT:
                        continue
                elif rarity == "prefer_rare":
                    if freq >= COMMON_CUT:
                        continue
                elif rarity == "prefer_common":
                    if freq <= RARE_CUT:
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
                # Random mode (Legacy)
                random.shuffle(stanza_clean)
                for kw in stanza_clean[:max_per_stanza]:
                    lw = kw.lower()
                    if lw in seen:
                        continue
                    seen.add(lw)
                    final_keywords.append(kw)
                    if len(final_keywords) >= request.max_words:
                        break
        
        # If LLM mode, now perform the selection
        if use_llm:
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

        # 5. Colors for each keyword (unchanged logic)
        final_suggestions: list[WordSuggestion] = []
        for word in final_keywords:
            try:
                color_data = get_colorspace_analysis(word, request.colorspace)
                final_suggestions.append(
                    WordSuggestion(word=word, colors=color_data)
                )
            except Exception as e:
                print(f"    - Error getting colors for '{word}': {e}")
                default_color = (
                    {"air": 1.0}
                    if request.colorspace == "elements"
                    else {"calm": 1.0}
                )
                final_suggestions.append(
                    WordSuggestion(word=word, colors=default_color)
                )
        # 6. Build semantic edges between suggestion words
        edges: list[EdgeInfo] = []

        try:
            if final_keywords:
                # query embedding is already computed as `query_embedding`
                q_vec = query_embedding  # shape (d,)

                # embeddings + similarity to the query context
                word_vecs = []
                sims_to_query = []
                for w in final_keywords:
                    v = EMBEDDING_MODEL.encode([w]).astype("float32")[0]
                    word_vecs.append(v)
                    sims_to_query.append(cosine_similarity(v, q_vec))

                n = len(final_keywords)
                for i in range(n):
                    for j in range(i + 1, n):
                        sim_ij = cosine_similarity(word_vecs[i], word_vecs[j])
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
        print(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))



CORPUS_DIR = "corpuses"  # or whatever you use now

@app.get("/api/corpora")
def list_corpora():
    """
    Return the distinct 'source' field values from DOCUMENTS.
    Example: {"corpora": ["Abram - The Spell of the Sensuous.txt", ...]}
    """
    try:
        sources = sorted({doc["source"] for doc in DOCUMENTS})
    except Exception:
        sources = []
    return {"corpora": sources}



# --- Run the Server ---
if __name__ == "__main__":
    print("Starting Uvicorn server on http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)