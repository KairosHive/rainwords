# RainWords — AI-Assisted Poetry

RainWords is a playful writing console that suggests semantically related words as glowing bubbles you can “catch” to weave into your poem.  


<img width="1902" height="951" alt="image" src="https://github.com/user-attachments/assets/f7e81b67-8a05-44ef-808c-623a9abcb6c5" />

---

## ✨ What’s inside
- **Interactive editor** (left) and **word-bubble field** (right)
- **Corpuses** picker (multi-select) to constrain suggestions
- **Parts-of-speech** filter (NOUN / ADJ / VERB)
- **Attention**: current line vs whole poem
- **Colorspaces** for visual mood (Elements / Temperature / Chakras)
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
python -m rainwords.convert_pdf
```

#### 4.2 Rebuild from text files
Put `.txt` source files in the `corpuses/` folder. Each file name becomes a **source label** (e.g., `Artaud - Le Théâtre Et Son Double.txt`). You can mix languages; the extractor auto-detects FR/EN.

```bash
python -m rainwords.corpus_builder
```

This script will:
- Read each `.txt` file in `corpuses/`
- Split into short stanzas / paragraphs
- Embed each chunk with `all-MiniLM-L6-v2`
- Save `poetry.index` (FAISS) and `poetry_docs.pkl` (metadata)

> **Tip:** keep chunks short (1–3 sentences). It improves suggestion quality.

---

## 🧠 How it works (short)

1. The API embeds the **current line** or **full poem** (your choice).
2. We retrieve the most similar stanzas from the corpus, **optionally filtered by selected corpora**.
3. We **extract keywords** from those stanzas.
4. Each keyword is mapped to a **colorspace** to style the bubble.
5. Bubbles are rendered; click a bubble to insert its word into the poem.

