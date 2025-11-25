import os
import re
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
from collections import Counter
import argparse  # NEW

# NEW: import your keyword extraction so frequencies match suggestion logic
from .semantics_and_colors import extract_keywords

# --- Configuration ---

MODEL_NAME = 'all-MiniLM-L6-v2'

BASE_DIR = Path(__file__).resolve().parent.parent

CORPUS_DIR = BASE_DIR / "corpuses"
INDEX_FILE = BASE_DIR / "poetry.index"
DOCS_FILE  = BASE_DIR / "poetry_docs.pkl"

# NEW: word-frequency cache file
WORD_FREQ_FILE = BASE_DIR / "word_freq.pkl"


# --- Word frequency computation ---

def compute_and_save_word_freq(documents, out_path: Path):
    """
    Build a global word frequency map from all chunk texts and
    save it (with rarity thresholds) to `out_path` as a pickle.
    """
    if not documents:
        print("No documents available to compute word frequencies.")
        return

    print("Computing global word frequencies from corpus...")
    counter = Counter()

    for doc in documents:
        text = doc.get("text", "")
        # align with suggestion logic: pos=None, max_per_chunk reasonably high
        kws = extract_keywords(text, lang=None, pos=None, max_per_chunk=100)
        for w in kws:
            counter[w.lower()] += 1

    freq_dict = dict(counter)
    if not freq_dict:
        print("Warning: no words collected for frequency map.")
        return

    # rarity thresholds
    freq_arr = np.array(list(freq_dict.values()), dtype=float)
    rare_cut   = float(np.percentile(freq_arr, 25))   # bottom 25% = "rare"
    common_cut = float(np.percentile(freq_arr, 75))   # top 25%   = "common"

    print(
        f"Word frequencies: min={freq_arr.min()}, max={freq_arr.max()}, "
        f"rare_cut={rare_cut}, common_cut={common_cut}, "
        f"unique_words={len(freq_dict)}"
    )

    payload = {
        "freq": freq_dict,
        "rare_cut": rare_cut,
        "common_cut": common_cut,
    }

    try:
        with open(out_path, "wb") as f:
            pickle.dump(payload, f)
        print(f"Saved word frequency cache → '{out_path}'.")
    except Exception as e:
        print(f"Error saving word frequency cache: {e}")


# --- Corpus loading & chunking ---


def _split_into_stanzas(content: str) -> list[str]:
    """Chunk by blank lines (current behavior)."""
    return re.split(r'\n\s*\n', content)


def _split_into_sentences(content: str) -> list[str]:
    """
    Very simple sentence splitter:
    split on . ! ? followed by whitespace + (optionally) capital letter.
    Works OK for FR/EN prose; for poetry it’s approximate, but usable.
    """
    # Normalize internal whitespace a bit first
    text = re.sub(r'\s+', ' ', content.strip())
    if not text:
        return []

    # Split keeping the delimiter attached to the sentence
    parts = re.split(r'([.!?])', text)
    sentences = []
    current = ""

    for part in parts:
        if not part:
            continue
        current += part
        if part in ".!?":  # end of sentence
            sent = current.strip()
            if sent:
                sentences.append(sent)
            current = ""

    # tail without terminal punctuation
    tail = current.strip()
    if tail:
        sentences.append(tail)

    return sentences


def load_and_chunk_corpus(dir_path: str, chunk_mode: str = "stanza"):
    """
    Load .txt files and chunk them either by stanza (blank lines)
    or by sentences, depending on `chunk_mode`:
        - "stanza": current behavior (paragraphs / stanzas)
        - "sentence": sentence-by-sentence
    """
    chunk_mode = chunk_mode.lower().strip()
    if chunk_mode not in ("stanza", "sentence"):
        print(f"Unknown chunk_mode '{chunk_mode}', falling back to 'stanza'.")
        chunk_mode = "stanza"

    documents = []
    
    if not os.path.exists(dir_path):
        print(f"Error: Corpus directory not found at '{dir_path}'")
        print("Please create it and add your .txt files.")
        return []

    print(f"Loading corpus from '{dir_path}' with chunking='{chunk_mode}'...")
    print("Files found in directory:")
    for entry in os.listdir(dir_path):
        print("  •", repr(entry))
    
    for filename in os.listdir(dir_path):
        if not filename.lower().endswith(".txt"):
            continue

        filepath = os.path.join(dir_path, filename)
        print(f"  - Processing {filename}...")
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            if chunk_mode == "stanza":
                chunks = _split_into_stanzas(content)
                chunk_type = "stanza"
            else:
                chunks = _split_into_sentences(content)
                chunk_type = "sentence"

            for chunk in chunks:
                # Clean up the chunk:
                # 1. Replace newlines inside chunk with a space
                # 2. Split by space to remove duplicated whitespace
                # 3. Join with a single space
                text = " ".join(chunk.split()).strip()

                # Filter out empty or very short chunks
                if text and len(text.split()) > 3:
                    documents.append({
                        "text": text,
                        "source": filename,
                        "type": chunk_type,
                    })

        except Exception as e:
            print(f"    - Error processing {filename}: {e}")

    total_docs = len(documents)
    if total_docs > 0:
        print(f"\nSuccessfully loaded and chunked {total_docs} {chunk_mode}s.")
    else:
        print("\nNo documents were loaded. Is the 'corpuses' folder empty?")
        
    return documents


# --- FAISS index building ---


def compute_and_build_index(documents, index_file_path, docs_file_path):
    """
    Takes the chunked documents, computes embeddings, builds a FAISS index,
    and saves the index and document map to disk.
    """
    if not documents:
        print("No documents to index. Exiting.")
        return

    # 1. Load the embedding model
    print(f"Loading embedding model '{MODEL_NAME}'...")
    try:
        model = SentenceTransformer(MODEL_NAME)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure you have an internet connection to download the model,")
        print("or that the model is cached locally.")
        return
        
    # 2. Get the text for embedding
    texts = [doc['text'] for doc in documents]
    
    # 3. Compute embeddings
    print(f"Computing {len(texts)} embeddings... (This may take a while on first run)")
    try:
        embeddings = model.encode(texts, show_progress_bar=True)
    except Exception as e:
        print(f"Error computing embeddings: {e}")
        return

    embeddings = embeddings.astype('float32')

    # 4. Build the FAISS index
    dimension = embeddings.shape[1]
    print(f"\nBuilding FAISS index (Dimension: {dimension})...")
    
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    print(f"Index built. Total vectors: {index.ntotal}")

    # 5. Save the index and the document map
    print(f"Saving FAISS index to '{index_file_path}'...")
    faiss.write_index(index, index_file_path)
    
    print(f"Saving document map to '{docs_file_path}'...")
    with open(docs_file_path, "wb") as f:
        pickle.dump(documents, f)
        
    print("\n--- Build Complete ---")
    print(f"Database created: {index_file_path}, {docs_file_path}")
    print("You can now run the RainWords server.")


# --- Run the Script ---


def main():
    """Entry point for the rainwords.corpus_builder CLI."""
    parser = argparse.ArgumentParser(
        description="Build RainWords corpus index from .txt files."
    )
    parser.add_argument(
        "--chunking",
        choices=["stanza", "sentence"],
        default="stanza",
        help="How to chunk the text: 'stanza' (blank-line paragraphs) or 'sentence'."
    )
    args = parser.parse_args()

    # 1. Load and chunk
    all_documents = load_and_chunk_corpus(str(CORPUS_DIR), chunk_mode=args.chunking)

    # 2. Embed and save
    if all_documents:
        compute_and_build_index(all_documents, str(INDEX_FILE), str(DOCS_FILE))
        # 3. compute and save word frequencies for rarity filter
        compute_and_save_word_freq(all_documents, WORD_FREQ_FILE)


if __name__ == "__main__":
    main()
