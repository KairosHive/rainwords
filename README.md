# RainWords — AI-Assisted Poetry

RainWords is a playful writing console that suggests semantically related words as glowing bubbles you can “catch” to weave into your poem.  


<img width="1917" height="983" alt="image" src="https://github.com/user-attachments/assets/f072df71-5923-4df1-b83f-0dd3e4a01167" />

---

## ✨ What’s inside
- **Interactive editor** (left) and **word-bubble field** (right)
- **Corpuses** picker (multi-select) to constrain suggestions
- **Parts-of-speech** filter (NOUN / ADJ / VERB)
- **Attention**: current line vs whole poem
- **Colorspaces** for visual mood (Elements / Temperature / Chakras)
- **Corpus map** (MAP button, opens `/map` in its own window). It follows the poem as you write. *Strips* show each corpus from first page to last, with the passages each line drew from lit up. *Atlas* places every passage on a 2D map of meaning and traces the poem's path across it.
- **LLM selection** with Gemini (default `gemini-3.5-flash-lite`, set `GEMINI_MODEL` to change it), a local Ollama model, or HuggingFace
- **Local-only** embeddings (`all-MiniLM-L6-v2`) + FAISS vector search

---

## 🚀 Quick start

### 1) Requirements
- Python ≥ 3.10
- Recommended: create a virtual env

```bash
# Create a conda env with Python >= 3.10 and activate it
conda create -n rainwords python=3.10 -y
conda activate rainwords

```

### 2. Install (editable)
```bash
pip install -e .
```
This installs a console command called **rainwords**.

### 3. Run RainWords

```bash
rainwords
```

This:

- starts the backend
- serves the UI
- automatically opens your browser at http://127.0.0.1:8000

### 4. Rebuild the database (optional)

#### 4.1 Convert PDFs to text (optional)
```bash
rainwords.convert_pdf
```

#### 4.2 Rebuild from text files
Put `.txt` source files in the `corpuses/` folder. Each file name becomes a **source label** (e.g., `Artaud - Le Théâtre Et Son Double.txt`). You can mix languages; the extractor auto-detects FR/EN.

```bash
rainwords.corpus_builder
```

This script will:
- Read each `.txt` file in `corpuses/`
- Split into short stanzas / paragraphs
- Embed each chunk with `all-MiniLM-L6-v2`
- Save `poetry.index` (FAISS) and `poetry_docs.pkl` (metadata)

> **Tip:** keep chunks short (1–3 sentences). It improves suggestion quality.

#### 4.3 Rebuild the corpus atlas
After rebuilding the index, lay the passages out again for the map window's Atlas view. This step needs `umap-learn` (`pip install umap-learn`), used only here.

```bash
python -m rainwords.build_corpus_atlas
```

It writes `rainwords/corpus_atlas.json`. Uploaded corpora are placed on the atlas automatically, next to their nearest built-in passages.

---

## 🧠 How it works (short)

1. The API embeds the **current line** or **full poem** (your choice).
2. We retrieve the most similar stanzas from the corpus, **optionally filtered by selected corpora**.
3. We **extract keywords** from those stanzas.
4. Each keyword is mapped to a **colorspace** to style the bubble.
5. Bubbles are rendered; click a bubble to insert its word into the poem.

## Resources

Find french books here : https://www.ebooksgratuits.com/ebooks.php
