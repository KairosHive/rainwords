import os
import nltk
import pickle
from sentence_transformers import SentenceTransformer

print("--- Preloading AI Models for Docker Build ---")

# 1. Download NLTK Data
print("Downloading NLTK data...")
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

# 2. Download Sentence Transformer Model
print("Downloading SentenceTransformer model (all-MiniLM-L6-v2)...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model downloaded and cached.")

# 3. Pre-compute Concept Embeddings
print("Pre-computing concept embeddings...")

MODE_KEYS = {
    "elements":         ["fire", "air", "water", "earth"],
    "temperature":      ["cold", "cool", "neutral", "warm", "hot"],
    "chakras":          ["root", "sacral", "solar", "heart", "throat", "third_eye", "crown"],
    "seasons":          ["spring", "summer", "autumn", "winter"],
    "emotions":         ["joy", "sadness", "anger", "fear", "surprise", "disgust"],
    "hermetic_alchemy": ["nigredo", "albedo", "citrinitas", "rubedo"],
    "directions":       ["north", "east", "south", "west", "center"],
}

def build_full_colorspace_labels():
    base_hues = [
        "red", "orange", "amber", "yellow", "chartreuse", "lime",
        "green", "emerald", "teal", "turquoise", "cyan", "sky blue",
        "blue", "indigo", "violet", "magenta", "fuchsia", "pink",
        "rose", "brown", "chocolate", "tan", "olive", "gold",
        "silver", "gray", "black", "white"
    ]
    tones = ["light", "medium", "dark"]
    qualifiers = ["cool", "warm"]
    labels = []
    for hue in base_hues:
        for tone in tones:
            for q in qualifiers:
                labels.append(f"{tone} {q} {hue}")
    return labels[:150]

FULL_COLOR_LABELS = build_full_colorspace_labels()

embeddings_cache = {}

# Encode standard modes
for mode, concepts in MODE_KEYS.items():
    print(f"Encoding {mode}...")
    embeddings_cache[mode] = model.encode(concepts, convert_to_tensor=True)

# Encode full colorspace
# print(f"Encoding full colorspace ({len(FULL_COLOR_LABELS)} items)...")
# embeddings_cache["full"] = model.encode(FULL_COLOR_LABELS, convert_to_tensor=True)

# Save to disk
output_path = os.path.join(os.path.dirname(__file__), "concept_embeddings.pkl")
with open(output_path, "wb") as f:
    pickle.dump(embeddings_cache, f)

print(f"Embeddings saved to {output_path}")
print("--- Preload Complete ---")
