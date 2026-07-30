import os
import pickle
import faiss
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

# Load .env file before importing embedder
env_path = Path(__file__).resolve().parent / ".env"
if env_path.exists():
    print(f"Loading .env from {env_path}")
    load_dotenv(env_path, override=True)
else:
    print(f"Warning: .env not found at {env_path}")

from .cloudflare_embedder import create_embedder
from .text_pipeline import chunk_text

# --- Configuration ---

MODEL_NAME = 'all-MiniLM-L6-v2'

BASE_DIR = Path(__file__).resolve().parent

CORPUS_DIR = BASE_DIR / "../corpuses"
INDEX_FILE = BASE_DIR / "poetry.index"
DOCS_FILE  = BASE_DIR / "poetry_docs.pkl"


# --- Main Functions ---


def load_and_chunk_corpus(dir_path):
    documents = []
    
    if not os.path.exists(dir_path):
        print(f"Error: Corpus directory not found at '{dir_path}'")
        print("Please create it and add your .txt files.")
        return []

    print(f"Loading corpus from '{dir_path}'...")
    print("Files found in directory:")
    for entry in os.listdir(dir_path):
        print("  •", repr(entry))
    
    for filename in os.listdir(dir_path):
        # Make the extension check case-insensitive
        if filename.lower().endswith(".txt"):
            filepath = os.path.join(dir_path, filename)
            print(f"  - Processing {filename}...")
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Split into stanzas via the shared chunker (same logic the
                # live upload endpoint uses).
                documents.extend(chunk_text(content, filename))

            except Exception as e:
                print(f"    - Error processing {filename}: {e}")

    total_docs = len(documents)
    if total_docs > 0:
        print(f"\nSuccessfully loaded and chunked {total_docs} stanzas.")
    else:
        print("\nNo documents were loaded. Is the 'corpus' folder empty?")
        
    return documents

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
        model = create_embedder(MODEL_NAME)
        model_dim = model.get_sentence_embedding_dimension()
        print(f"✓ Embedding model loaded. Dimension: {model_dim}")
        print(f"  Model type: {type(model).__name__}")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure you have an internet connection to download the model,")
        print("or that the model is cached locally.")
        return
        
    # 2. Get the text for embedding
    texts = [doc['text'] for doc in documents]
    
    # 3. Compute embeddings in batches
    print(f"Computing {len(texts)} embeddings in batches...")
    print("(This may take a while, especially with API calls)")
    
    try:
        # Use batch_size for efficient API calls (especially for Cloudflare)
        batch_size = 150  # Cloudflare can handle up to 100 texts per request
        all_embeddings = []
        
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            print(f"  Processing batch {batch_num}/{total_batches} ({len(batch_texts)} texts)...")
            
            batch_embeddings = model.encode(batch_texts)
            all_embeddings.append(batch_embeddings)
        
        embeddings = np.vstack(all_embeddings)
        print(f"✓ All embeddings computed ({embeddings.shape})")
        
    except Exception as e:
        print(f"Error computing embeddings: {e}")
        return

    # FAISS requires float32.
    embeddings = embeddings.astype('float32')

    # 4. Build the FAISS index
    dimension = embeddings.shape[1]
    print(f"\nBuilding FAISS index (Dimension: {dimension})...")
    
    # We use IndexFlatL2, a simple and effective index for exact search.
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
    print("You can now run the 'main.py' server.")

# --- Run the Script ---

def main():
    """Entry point for the rainwords.corpus_builder CLI."""
    # 1. Load and chunk
    all_documents = load_and_chunk_corpus(str(CORPUS_DIR))

    # 2. Embed and save
    if all_documents:
        compute_and_build_index(all_documents, str(INDEX_FILE), str(DOCS_FILE))



if __name__ == "__main__":
    main()
