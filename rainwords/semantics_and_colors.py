import os
import re
import json
import nltk
from nltk.corpus import stopwords
import pickle
from typing import List, Dict, Set
from functools import lru_cache
import string
from typing import Iterable, List, Optional
from collections import Counter

from sentence_transformers import SentenceTransformer, util

try:
    import spacy
    _SPACY_OK = True
except Exception:
    _SPACY_OK = False
    spacy = None
    
# Load model once
try:
    nlp_fr = spacy.load("fr_core_news_md")
except OSError:
    nlp_fr = spacy.load("fr_core_news_sm")
    
try:
    nlp_en = spacy.load("en_core_web_sm")
except Exception:
    nlp_en = None

POS_EXPANSION = {
    "NOUN": {"NOUN"},
    "VERB": {"VERB"},
    "ADJ":  {"ADJ"},
}

def _expand_pos(requested):
    if not requested:
        return None
    out = set()
    for r in requested:
        out |= POS_EXPANSION.get(r, {r})
    return out

# --- Local AI Configuration ---

try:
    print("Loading local model for Semantics and Colors analysis...")
    LOCAL_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    print("Local AI model loaded.")
except Exception as e:
    print(f"Warning: Could not load local AI model for color analysis. Error: {e}")
    LOCAL_MODEL = None

# Lightweight French stopword fallback (augment NLTK’s if needed)
_FALLBACK_FR_STOPS = {
    "le","la","les","de","des","du","un","une","et","ou","mais","donc","or","ni","car",
    "à","au","aux","en","dans","sur","sous","chez","vers","par","pour","avec","sans","entre",
    "se","sa","son","ses","leurs","leur","nos","notre","votre","vos","mon","ma","mes",
    "ce","cet","cette","ces","cela","ça","c’","d’","l’","qu’","que","qui","quoi","où",
    "ne","pas","plus","moins","tres","très","comme","ainsi","alors","si","quand","puis",
    "y","en","on","nous","vous","ils","elles","il","elle","je","tu","me","te","moi","toi",
    "être","avoir","fait","faites","faites-le"
}

_WORD_RE = re.compile(r"\b[\w’']+\b", re.UNICODE)
# --- Colorspace Concept Definitions ---

# keys used consistently everywhere (embedding + vectors)
MODE_KEYS = {
    "elements":         ["fire", "air", "water", "earth"],
    "temperature":      ["cold", "cool", "neutral", "warm", "hot"],
    "chakras":          ["root", "sacral", "solar", "heart", "throat", "third_eye", "crown"],
    "seasons":          ["spring", "summer", "autumn", "winter"],
    "emotions":         ["joy", "sadness", "anger", "fear", "surprise", "disgust"],
    "hermetic_alchemy": ["nigredo", "albedo", "citrinitas", "rubedo"],
    "directions":       ["north", "east", "south", "west", "center"],
}


ELEMENT_CONCEPTS     = MODE_KEYS["elements"]
TEMPERATURE_CONCEPTS = MODE_KEYS["temperature"]
SEASONS_CONCEPTS     = MODE_KEYS["seasons"]
EMOTIONS_CONCEPTS    = MODE_KEYS["emotions"]
HERMETIC_ALCHEMY_CONCEPTS = MODE_KEYS["hermetic_alchemy"]
DIRECTIONS_CONCEPTS  = MODE_KEYS["directions"]

# make this consistent with MODE_KEYS!
CHAKRA_CONCEPTS = MODE_KEYS["chakras"]



def _language_stopwords(lang: Optional[str]) -> set:
    try:
        if lang == "fr":
            sw = set(stopwords.words("french"))
            sw |= _FALLBACK_FR_STOPS
            return sw
        elif lang == "en":
            return set(stopwords.words("english"))
    except Exception:
        pass
    return set()

# 4) Full Colorspace – generate ~500 distinct color phrases
def build_full_colorspace_labels() -> List[str]:
    """
    Builds a list of 500-ish distinct color descriptors spanning
    a wide semantic range, e.g. "very light cool blue",
    "deep warm red", etc.
    """
    base_hues = [
        "red", "orange", "amber", "yellow", "chartreuse", "lime",
        "green", "emerald", "teal", "turquoise", "cyan", "sky blue",
        "blue", "indigo", "violet", "magenta", "fuchsia", "pink",
        "rose", "brown", "chocolate", "tan", "olive", "gold",
        "silver", "gray", "black", "white"
    ]
    tones = [
        "very light", "light", "soft", "pastel", "muted",
        "medium", "rich", "deep", "dark", "very dark"
    ]
    qualifiers = ["cool", "warm", "bright"]

    labels = []
    for hue in base_hues:
        for tone in tones:
            for q in qualifiers:
                labels.append(f"{tone} {q} {hue}")
    # Trim to exactly 500 (deterministic order)
    return labels[:500]

FULL_COLOR_LABELS = build_full_colorspace_labels()

# --- Pre-compute Concept Embeddings ---

if LOCAL_MODEL:
    print("Pre-computing concept embeddings...")

    ELEMENT_EMBEDDINGS = LOCAL_MODEL.encode(
        ELEMENT_CONCEPTS, convert_to_tensor=True
    )
    TEMPERATURE_EMBEDDINGS = LOCAL_MODEL.encode(
        TEMPERATURE_CONCEPTS, convert_to_tensor=True
    )
    CHAKRA_EMBEDDINGS = LOCAL_MODEL.encode(
        CHAKRA_CONCEPTS, convert_to_tensor=True
    )
    SEASONS_EMBEDDINGS = LOCAL_MODEL.encode(
        SEASONS_CONCEPTS, convert_to_tensor=True
    )
    EMOTIONS_EMBEDDINGS = LOCAL_MODEL.encode(
        EMOTIONS_CONCEPTS, convert_to_tensor=True
    )
    HERMETIC_ALCHEMY_EMBEDDINGS = LOCAL_MODEL.encode(
        HERMETIC_ALCHEMY_CONCEPTS, convert_to_tensor=True
    )
    DIRECTIONS_EMBEDDINGS = LOCAL_MODEL.encode(
        DIRECTIONS_CONCEPTS, convert_to_tensor=True
    )
    
    FULL_COLOR_EMBEDDINGS = LOCAL_MODEL.encode(
        FULL_COLOR_LABELS, convert_to_tensor=True
    )

    print("Concept embeddings are ready.")
else:
    ELEMENT_EMBEDDINGS = None
    TEMPERATURE_EMBEDDINGS = None
    CHAKRA_EMBEDDINGS = None
    FULL_COLOR_EMBEDDINGS = None


LETTER_CLASS = r"A-Za-zÀ-ÖØ-öø-ÿ"
WORD_FORM_RE = re.compile(
    rf"^[{LETTER_CLASS}][{LETTER_CLASS}'’\-]*[{LETTER_CLASS}]$",
    flags=re.UNICODE,
)

def is_good_word_form(text: str) -> bool:
    """
    Accept only well-formed word tokens:
    - at least 3 characters
    - no digits
    - only letters (incl. accents) + internal ' ’ - 
    - must start & end with a letter (no leading/trailing hyphen/apostrophe)
    """
    text = (text or "").strip().lower()

    if len(text) < 3:
        return False

    # reject any digits
    if any(ch.isdigit() for ch in text):
        return False

    # must match our word shape
    if not WORD_FORM_RE.match(text):
        return False

    # extra safety: avoid dangling punctuation
    if text.endswith(("-", "'", "’")):
        return False

    return True

# --- replace your extract_keywords completely with this ---
def extract_keywords(
    text: str,
    lang: str | None = None,
    pos: list[str] | None = None,
    max_per_chunk: int = 50,
) -> list[str]:
    if not text or not text.strip():
        return []

    # crude FR/EN detection
    if lang is None:
        lang = "fr" if any(c in text for c in "éèàçùôîïâêœ") else "en"

    wanted = _expand_pos(pos)

    # -------- FRENCH: surface form (no lemma) --------
    if lang == "fr":
        doc = nlp_fr(text)
        bag: list[str] = []

        for t in doc:
            if t.is_space or t.is_punct or t.is_stop:
                continue
            if wanted is not None and t.pos_ not in wanted:
                continue

            surface = t.text.lower()
            if not is_good_word_form(surface):
                continue

            bag.append(surface)

        return [w for w, _ in Counter(bag).most_common(max_per_chunk)]

    # -------- ENGLISH with spaCy (if available) --------
    if lang == "en" and nlp_en is not None:
        doc = nlp_en(text)
        bag: list[str] = []

        for t in doc:
            if t.is_space or t.is_punct or t.is_stop:
                continue
            if wanted is not None and t.pos_ not in wanted:
                continue

            # you can use lemma or surface; lemma is usually fine in EN
            form = t.lemma_.lower()
            if not is_good_word_form(form):
                continue

            bag.append(form)

        return [w for w, _ in Counter(bag).most_common(max_per_chunk)]

    # -------- ENGLISH fallback: NLTK tagger --------
    tokens = re.findall(r"\b[a-zA-Z]{3,}\b", text)
    PTB2U = {
        "NN": "NOUN", "NNS": "NOUN", "NNP": "NOUN", "NNPS": "NOUN",
        "JJ": "ADJ", "JJR": "ADJ", "JJS": "ADJ",
        "VB": "VERB", "VBD": "VERB", "VBG": "VERB",
        "VBN": "VERB", "VBP": "VERB", "VBZ": "VERB",
        "RB": "ADV", "RBR": "ADV", "RBS": "ADV",
    }
    sw = set(stopwords.words("english"))
    try:
        tagged = nltk.pos_tag(tokens)
    except Exception:
        tagged = [(t, None) for t in tokens]

    bag: list[str] = []
    for tok, ptb in tagged:
        upos = PTB2U.get(ptb)
        tl = tok.lower()
        if tl in sw:
            continue
        if wanted is not None and upos not in wanted:
            continue
        if not is_good_word_form(tl):
            continue
        bag.append(tl)

    return [w for w, _ in Counter(bag).most_common(max_per_chunk)]



# semantics_and_colors.py (add near imports)
from typing import Tuple

def _guess_lang_stopword_overlap(tokens_lower: list[str]) -> str | None:
    """Cheap EN/FR detector: pick the language with higher stopword overlap."""
    try:
        en_sw = set(stopwords.words("english"))
    except Exception:
        en_sw = set()
    try:
        fr_sw = set(stopwords.words("french")) | _FALLBACK_FR_STOPS
    except Exception:
        fr_sw = set()

    if not en_sw and not fr_sw:
        return None

    tokset = set(tokens_lower)
    en_hits = len(tokset & en_sw)
    fr_hits = len(tokset & fr_sw)

    if en_hits == 0 and fr_hits == 0:
        return None
    if fr_hits > en_hits:
        return "fr"
    if en_hits > fr_hits:
        return "en"
    return None  # tie → unknown


# --- Colorspace Analysis ---

def _fallback_for_mode(norm_mode: str) -> Dict[str, float]:
    """
    Default fallback distribution if the model / embeddings are unavailable
    or similarities are all zero.
    """
    if norm_mode == "elements":
        return {"air": 1.0}
    elif norm_mode == "temperature":
        return {"neutral": 1.0}
    elif norm_mode == "chakras":
        return {"heart": 1.0}
    elif norm_mode == "full":
        # Pick a neutral-ish label in the full colorspace
        return {FULL_COLOR_LABELS[0]: 1.0} if FULL_COLOR_LABELS else {"neutral": 1.0}
    # generic fallback
    return {"air": 1.0}


@lru_cache(maxsize=500)
def get_colorspace_analysis(word: str, colorspace_mode: str) -> Dict[str, float]:
    """
    Map a word into one of several semantic colorspaces using local embeddings.

    colorspace_mode (case-insensitive):
        - "elements"   -> fire, air, water, earth
        - "temperature"-> cold, cool, neutral, warm, hot
        - "chakras"    -> 7 chakra centers
        - "seasons"    -> spring, summer, autumn, winter
        - "emotions"   -> joy, sadness, anger, fear, surprise, disgust
        - "Hermetic Alchemy" -> nigredo, albedo, citrinitas, rubedo
        - "directions" -> north, east, south, west, center
        - "full"       -> 500 fine-grained color phrases
    """
    # Normalize mode string
    norm_mode = colorspace_mode.lower().strip().replace(" ", "_")

    if not LOCAL_MODEL:
        return _fallback_for_mode(norm_mode)

    # Select concepts and their embeddings
    if norm_mode == "elements":
        concepts = ELEMENT_CONCEPTS
        concept_embeddings = ELEMENT_EMBEDDINGS
    elif norm_mode == "temperature":
        concepts = TEMPERATURE_CONCEPTS
        concept_embeddings = TEMPERATURE_EMBEDDINGS
    elif norm_mode == "chakras":
        concepts = CHAKRA_CONCEPTS
        concept_embeddings = CHAKRA_EMBEDDINGS
    elif norm_mode == "seasons":
        concepts = SEASONS_CONCEPTS
        concept_embeddings = SEASONS_EMBEDDINGS
    elif norm_mode == "emotions":
        concepts = EMOTIONS_CONCEPTS
        concept_embeddings = EMOTIONS_EMBEDDINGS
    elif norm_mode == "hermetic_alchemy":
        concepts = HERMETIC_ALCHEMY_CONCEPTS
        concept_embeddings = HERMETIC_ALCHEMY_EMBEDDINGS
    elif norm_mode == "directions":
        concepts = DIRECTIONS_CONCEPTS
        concept_embeddings = DIRECTIONS_EMBEDDINGS
    elif norm_mode in ("full", "full_colorspace", "full_color"):
        concepts = FULL_COLOR_LABELS
        concept_embeddings = FULL_COLOR_EMBEDDINGS
    else:
        # Unknown mode → default to elements
        concepts = ELEMENT_CONCEPTS
        concept_embeddings = ELEMENT_EMBEDDINGS
        norm_mode = "elements"

    if concept_embeddings is None:
        return _fallback_for_mode(norm_mode)

    try:
        # Encode the word
        word_embedding = LOCAL_MODEL.encode([word], convert_to_tensor=True)

        # Cosine similarity to each concept
        similarities = util.cos_sim(word_embedding, concept_embeddings)[0]
        scores = similarities.tolist()

        # 1) Clamp negatives
        positive_scores = [max(0.0, s) for s in scores]

        if sum(positive_scores) == 0:
            return _fallback_for_mode(norm_mode)

        # 2) Sharpen distribution (emphasize the most similar concept(s))
        gamma = 4  # adjust between ~1.5–3.0 for more/less contrast
        sharpened = [s ** gamma for s in positive_scores]

        total = sum(sharpened)
        normalized_scores = [s / total for s in sharpened]

        return dict(zip(concepts, normalized_scores))

    except Exception as e:
        print(f"Error in local analysis for word '{word}' (mode='{colorspace_mode}'): {e}")
        return _fallback_for_mode(norm_mode)
