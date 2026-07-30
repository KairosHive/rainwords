"""
Shared text-cleaning + chunking pipeline.

Used by both the offline CLIs (convert_pdf, corpus_builder) and the live
upload endpoint so that a PDF/.txt uploaded through the web app is sanitized
and chunked exactly like the corpora that shipped with RainWords.
"""
import re
from collections import Counter
from typing import List, Dict, Union, IO

# Letters incl. Latin-1 accented range (FR/EN).
LETTERS = r"A-Za-zÀ-ÖØ-öø-ÿ"

# Frequent short words we must never mistake for a running-header token.
_COMMON_WORDS = {
    "le", "la", "les", "un", "une", "des", "du", "de", "et", "ou", "au", "aux",
    "dans", "en", "ce", "cet", "cette", "ces", "son", "sa", "ses", "leur", "leurs",
    "qui", "que", "quoi", "pour", "par", "sur", "sous", "avec", "sans", "mais",
    "il", "elle", "ils", "elles", "je", "tu", "nous", "vous", "on", "ne", "pas",
    "the", "and", "of", "to", "in", "a", "an", "is", "it", "he", "she", "they",
    "we", "you", "for", "with", "as", "at", "by", "or", "but", "not", "this", "that",
}

# Publisher / license / URL boilerplate that marks a non-body chunk.
_BOILERPLATE_RE = re.compile(
    r"(www\.|https?://|©|\bISBN\b|project\s+gutenberg|ebooks?\s*france"
    r"|e-?books?\s*gratuits|copyright|tous droits r[ée]serv[ée]s|\b[ée]ditions?\b"
    r"|veuillez\s+[ée]crire|d[ée]p[ôo]t\s+l[ée]gal|achev[ée]\s+d['’]imprimer"
    r"|[\w.+-]+@[\w-]+\.[a-z]{2,}"
    # ebooksgratuits.com back-matter solicitation
    r"|votre\s+aide\s+est\s+la\s+bienvenue|vous\s+pouvez\s+nous\s+aider"
    r"|faire\s+conna[îi]tre\s+ces|classiques\s+litt[ée]raires"
    r"|cette\s+[ée]dition\s+[ée]lectronique|groupe\s+de\s+b[ée]n[ée]voles)",
    re.IGNORECASE,
)


def normalize_basic(text: str) -> str:
    """Normalize whitespace, smart quotes and dashes."""
    text = text.replace(' ', ' ')   # non-breaking space
    text = text.replace('\t', ' ')

    # Normalize quotes and dashes
    text = text.replace('“', '"').replace('”', '"')   # “ ”
    text = text.replace('’', "'").replace('‘', "'")   # ’ ‘
    text = text.replace('–', '-').replace('—', '-')   # – —

    return text


def strip_running_headers(text: str, min_repeat: int = 6) -> str:
    """
    Remove running page headers/footers (e.g. a book title repeated on every
    page, or a 'Poésies SONNET 40' footer with a page number).

    Two passes, so it works whether the header sits on its own line (fresh PDF
    extraction) or has already been merged into paragraph text (re-processing a
    stored .txt):

      A) drop short lines that repeat across the document;
      B) detect a capitalized phrase that recurs immediately before a page
         number ("Poésies SONNET 12") and strip that phrase+number everywhere;
         also strip its lone title word when that word appears almost only as
         part of the header (so real poem titles like "SONNET" are kept).
    """
    if not text:
        return text

    # --- Pass A: repeated whole short lines (raw extraction) ---
    lines = text.split("\n")
    freq = Counter(s for s in (ln.strip() for ln in lines) if s)
    header_lines = {
        s for s, c in freq.items()
        if c >= min_repeat and len(s) <= 60 and len(s.split()) <= 6
    }
    if header_lines:
        lines = [ln for ln in lines if ln.strip() not in header_lines]
        text = "\n".join(lines)

    # --- Pass B: detect a running header/footer *token* (e.g. the book title
    #     "Poésies") and strip it. A header token is a capitalized, non-common
    #     word that recurs and is USUALLY followed by an all-caps section word
    #     or a page number — which distinguishes it from ordinary capitalized
    #     verse words ("Mort", "Azur") and real poem titles ("SONNET").
    cap = r"[A-ZÀ-Ö][\wÀ-ÖØ-öø-ÿ’'\-]+"
    allcaps_or_num = re.compile(r"^(?:[A-ZÀ-Ö]{2,}|\d{1,3})$")
    token_total = Counter()
    token_header = Counter()
    for m in re.finditer(rf"\b({cap})\b([ \t]+(\S+))?", text):
        tok = m.group(1)
        token_total[tok] += 1
        nxt = (m.group(3) or "").strip(".,;:!?()[]«»\"'")
        if allcaps_or_num.match(nxt):
            token_header[tok] += 1

    for tok, total in token_total.items():
        if total < min_repeat or tok.lower() in _COMMON_WORDS or len(tok) < 4:
            continue
        if tok.isupper():
            continue   # ALL-CAPS tokens are poem titles / section headings — keep
        if token_header[tok] / total < 0.5:
            continue   # not usually followed by a section word / page number
        # A running header also bleeds into the middle of lines (after a
        # lowercase word); a real Titlecase word ("Mort") does not do this often.
        mid = len(re.findall(rf"[a-zà-öø-ÿ]\s+{re.escape(tok)}\b", text))
        if mid < 2:
            continue
        # 1) numbered footer form: "Header [Section] 12" (page number confirms it)
        text = re.sub(rf"\b{re.escape(tok)}\b(?:\s+{cap}){{0,2}}\s+\d{{1,3}}\b", " ", text)
        # 2) any remaining lone header token
        text = re.sub(rf"\b{re.escape(tok)}\b", " ", text)

    # --- Pass C: page number glued right after an ALL-CAPS title/section word,
    #     e.g. "CANTIQUE DE SAINT JEAN 19 Incandescent" -> "... JEAN Incandescent".
    #     Requires the number to be followed by resuming (capitalized) text, so
    #     real dates like "2 novembre 1877" (4-digit year / lowercase month) are
    #     left untouched.
    text = re.sub(r"\b([A-ZÀ-Ö]{2,})\s+\d{1,3}\b(?=\s+[A-ZÀ-Ö0-9])", r"\1", text)

    # tidy the whitespace the substitutions may have left behind
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_text(raw_text: str) -> str:
    """
    Clean and lightly reformat extracted PDF text:
    - normalize quotes/dashes/whitespace
    - de-hyphenate words broken across lines
    - remove pure page-number / roman-numeral lines
    - strip running page headers/footers
    - rebuild paragraph breaks into double newlines

    The output is blank-line-separated paragraphs, which is exactly what
    `chunk_text` expects.
    """
    # 0. Remove soft hyphens (line-breaking artifacts)
    raw_text = raw_text.replace('­', '')

    # NEW: normalize quotes/dashes/spaces up front (previously dead code).
    raw_text = normalize_basic(raw_text)

    # 1. De-hyphenate words broken across lines: "germ-\née" -> "germée"
    raw_text = re.sub(
        rf'([{LETTERS}])-\n([{LETTERS}])',
        r'\1\2',
        raw_text,
    )

    # 2. Split into lines to filter junk
    lines = raw_text.split('\n')
    cleaned_lines = []

    for line in lines:
        stripped = line.strip()

        if not stripped:
            continue

        if re.fullmatch(r'\d+', stripped):
            continue
        if re.fullmatch(r'[ivxlcdmIVXLCDM]+', stripped):
            continue

        cleaned_lines.append(stripped)

    text = "\n".join(cleaned_lines)

    # 3. Paragraph logic
    text = re.sub(r'(\n\s*){2,}', '<<PARAGRAPH_BREAK>>', text)
    text = re.sub(r'([.?!"])\n', r'\1<<PARAGRAPH_BREAK>>', text)
    text = re.sub(r'\n', ' ', text)

    text = re.sub(r'\s*<<PARAGRAPH_BREAK>>\s*', '\n\n', text).strip()

    # 4. Strip running page headers/footers (book title, "SONNET 40" page markers…)
    text = strip_running_headers(text)
    return text


def is_junk_chunk(text: str) -> bool:
    """
    True for non-body chunks: tables of contents and publisher/license/URL
    boilerplate that survive the length/alpha heuristics but aren't real text.
    """
    # Table of contents: many bullet separators or dotted leaders.
    if text.count("•") >= 3:
        return True
    if text.count("....") >= 2:
        return True
    # Publisher / license / URL / email boilerplate.
    if _BOILERPLATE_RE.search(text):
        return True
    return False


def is_good_stanza(text: str) -> bool:
    """Heuristic filter to drop junk chunks (headers, page furniture, etc.)."""
    words = text.split()
    if len(words) <= 3:
        return False

    # Require at least N alphabetic characters
    alpha = sum(ch.isalpha() for ch in text)
    if alpha < 20:
        return False

    # If too many digits / punctuation, drop
    non_alpha = sum(not ch.isalpha() and not ch.isspace() for ch in text)
    if non_alpha > alpha:
        return False

    return True


def chunk_text(content: str, source: str) -> List[Dict]:
    """
    Split blank-line-separated content into stanza documents.

    Returns a list of {"text", "source", "type"} dicts, matching the schema
    used by corpus_builder / poetry_docs.pkl.
    """
    documents: List[Dict] = []

    # Split the text by one or more empty lines (stanzas)
    stanzas = re.split(r'\n\s*\n', content)

    for stanza in stanzas:
        # 1. collapse newlines within a stanza to spaces
        # 2. remove duplicate whitespace
        text = " ".join(stanza.split()).strip()
        if is_good_stanza(text) and not is_junk_chunk(text):
            documents.append({
                "text": text,
                "source": source,
                "type": "stanza",
            })

    return documents


def extract_pdf_text(pdf: Union[str, IO[bytes]]) -> str:
    """
    Extract raw text from a PDF path or binary stream using pypdf.
    Returns the concatenated per-page text (before `clean_text`).
    """
    from pypdf import PdfReader

    reader = PdfReader(pdf)
    pages: List[str] = []
    for page in reader.pages:
        try:
            t = page.extract_text()
        except Exception:
            t = None
        if t:
            pages.append(t)
    return "\n".join(pages)
