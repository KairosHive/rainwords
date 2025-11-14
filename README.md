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

### 2) Install Python deps

```bash
pip install -r requirements.txt
```

### 3) Corpora
Put `.txt` source files in the `corpuses/` folder. Each file name becomes a **source label** (e.g., `Artaud - Le Théâtre Et Son Double.txt`). You can mix languages; the extractor auto-detects FR/EN.

A ready-made database is **already provided** in this repo:
```
poetry.index         # FAISS index
poetry_docs.pkl     # stanza map with "text" and "source"
```
You can use it directly, or rebuild from your own texts.

### 4) Run the API

```bash
python main.py
```
You should see:
```
Starting Uvicorn server on http://127.0.0.1:8000
...
RainWords API is running.
```

### 6) Open the frontend
Just open `index.html` in your browser (double-click or drag into a tab).  
When you press **Enter** in the editor, the page will call `POST /api/suggestions` and spawn new bubbles.

### 7) (Optional) Build the database from text files

If you want to rebuild the vector DB from your `corpuses/` folder, run:

```bash
python build_index.py
```
This script will:
- Read each `.txt` file in `corpuses/`
- Split into short stanzas / paragraphs
- Embed each chunk with `all-MiniLM-L6-v2`
- Save `poetry.index` (FAISS) and `poetry_docs.pkl` (metadata)

> **Tip:** keep chunks short (1–3 sentences). It improves suggestion quality.

**Convert PDFs to text** (optional)  
If you have PDF sources, run:
```bash
python convert_pdf.py
```
This will extract text from PDFs in the `corpuses/` folder and create corresponding `.txt` files.
---

## 🧠 How it works (short)

1. The API embeds the **current line** or **full poem** (your choice).
2. We retrieve the most similar stanzas from the corpus, **optionally filtered by selected corpora**.
3. We **extract keywords** from those stanzas.
4. Each keyword is mapped to a **colorspace** to style the bubble.
5. Bubbles are rendered; click a bubble to insert its word into the poem.

