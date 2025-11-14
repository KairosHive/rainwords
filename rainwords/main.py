import uvicorn
import faiss
import pickle
import numpy as np
import nltk
import re # Import re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import random
import os

# Import our helper functions
from semantics_and_colors import get_colorspace_analysis, extract_keywords

# --- Configuration ---
INDEX_FILE = "poetry.index"
DOCS_FILE = "poetry_docs.pkl"
MODEL_NAME = 'all-MiniLM-L6-v2'

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
except Exception as e:
    print(f"FATAL: Could not load embedding model. Error: {e}")
    exit()

print(f"Loading vector database from '{INDEX_FILE}'...")
try:
    VECTOR_INDEX = faiss.read_index(INDEX_FILE)
    print(f"FAISS index loaded. Total vectors: {VECTOR_INDEX.ntotal}")
except Exception as e:
    print(f"FATAL: Could not load FAISS index. Did you run 'build_index.py'? Error: {e}")
    exit()

print(f"Loading document map from '{DOCS_FILE}'...")
try:
    with open(DOCS_FILE, "rb") as f:
        DOCUMENTS = pickle.load(f)
    print(f"Document map loaded. Total documents: {len(DOCUMENTS)}")
    # Debug: show distinct corpus sources in the DB
    sources = sorted({doc["source"] for doc in DOCUMENTS})
    print("Available corpus sources in DOCUMENTS:")
    for s in sources:
        print("  •", repr(s))
except Exception as e:
    print(f"FATAL: Could not load document map. Did you run 'build_index.py'? Error: {e}")
    exit()

# --- Initialize FastAPI App ---
app = FastAPI(title="RainWords AI API")

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
    colorspace: str
    attention: str
    k: int = 5
    max_words: int = 10
    corpus: str | None = None            # legacy single
    corpora: list[str] | None = None     # NEW: multi
    pos: List[str] | None = None  # keep POS control


# This defines the "word" object we'll send back
class WordSuggestion(BaseModel):
    word: str
    colors: Dict[str, float]

# --- API Endpoints ---

@app.get("/")
def read_root():
    """ A simple endpoint to check if the server is running. """
    return {"status": "RainWords API is running."}


@app.post("/api/suggestions", response_model=List[WordSuggestion])
def get_suggestions(request: SuggestionRequest): # <-- Removed 'async'
    """
    The main endpoint for getting AI-powered word suggestions.
    """
    print(f"\nReceived suggestion request:")
    print(f"  - Colorspace: {request.colorspace}")
    print(f"  - Attention: {request.attention}")
    print(f"  - Corpus: {repr(request.corpus)}")

    try:
        # 1. Get Query Text based on "attention"
        full_text = request.text.strip()
        if not full_text:
            return [] # Return empty list if there's no text

        if request.attention == 'line':
            # Get the last non-empty line
            query_text_lines = [line for line in full_text.split('\n') if line.strip()]
            if not query_text_lines: # Handle case where text is just whitespace
                return []
            query_text = query_text_lines[-1]
        else:
            query_text = full_text
        
        print(f"  - Querying with text: \"{query_text[:50]}...\"")

        # 2. Compute embedding for the query text
        # 2. Compute embedding for the query text
        query_embedding = EMBEDDING_MODEL.encode([query_text]).astype('float32')

        # 3. Search the Vector Database
        #    If a corpus is selected, search deeper so we have enough candidates
        # before VECTOR_INDEX.search
        if request.corpus:
            search_k = min(VECTOR_INDEX.ntotal, max(request.k * 20, 100))
        else:
            search_k = min(VECTOR_INDEX.ntotal, max(request.k * 10, 80))
        D, I = VECTOR_INDEX.search(query_embedding, k=search_k)

        all_indices = list(I[0])

        # Build allowed sources from multi OR single
        allowed_sources = None
        if request.corpora:                       # multi
            allowed_sources = {s.lower() for s in request.corpora}
        elif request.corpus:                       # single
            allowed_sources = {request.corpus.lower()}

        # Filter by allowed sources if provided
        if allowed_sources:
            retrieved_indices = [
                idx for idx in all_indices
                if DOCUMENTS[idx]["source"].lower() in allowed_sources
            ][:max(request.k, request.max_words * 3)]
        else:
            retrieved_indices = all_indices[:max(request.k, request.max_words * 3)]

        print("Allowed sources:", allowed_sources if allowed_sources else "(ALL)")
        print("Sample of DOCUMENTS sources:", sorted({doc["source"] for doc in DOCUMENTS})[:8])



        # 4. Retrieve & Extract ONE RANDOM keyword per stanza
        print("\n--- Retrieved Stanzas (in similarity order) ---")
        for rank, idx in enumerate(retrieved_indices):
            stanza_text = DOCUMENTS[idx]['text']
            print(f"[{rank+1}] (id={idx}): {stanza_text}")
        print("------------------------------------------------\n")

        # --- old: one-per-stanza loop ... replace with pooled selection ---
        user_words = set(re.findall(r'\b\w+\b', full_text.lower()))
        pooled = []

        for idx in retrieved_indices:
            stanza_text = DOCUMENTS[idx]['text']
            stanza_keywords = list(extract_keywords(
                stanza_text,
                lang=None,              # auto-detect (FR→spaCy)
                pos=request.pos         # ["NOUN","ADJ","VERB"] or None
            ))

            for kw in stanza_keywords:
                lw = kw.lower()
                if lw in user_words:
                    continue
                pooled.append(kw)

        # de-duplicate (case-insensitive) while preserving order
        seen = set()
        pooled = [w for w in pooled if not (w.lower() in seen or seen.add(w.lower()))]

        # randomize and clip to the requested amount
        random.shuffle(pooled)
        final_keywords = pooled[:request.max_words]
        print(f"  - Pooled {len(pooled)} candidates → returning {len(final_keywords)}")


        print(f"  - Extracted {len(final_keywords)} new keywords: {final_keywords}")


        # 6. Get Colorspace Analysis for each keyword
        final_suggestions = []
        for word in final_keywords:
            try:
                # (UPDATED) This is no longer an async call
                color_data = get_colorspace_analysis(word, request.colorspace) # <-- Removed 'await'
                final_suggestions.append(WordSuggestion(word=word, colors=color_data))
            except Exception as e:
                print(f"    - Error getting colors for '{word}': {e}")
                # Add it anyway with a default color
                default_color = {"air": 1.0} if request.colorspace == "elements" else {"calm": 1.0}
                final_suggestions.append(WordSuggestion(word=word, colors=default_color))

        print(f"  - Returning {len(final_suggestions)} suggestions to frontend.")
        return final_suggestions

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