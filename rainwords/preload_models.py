import os
import nltk
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

print("--- Preload Complete ---")
