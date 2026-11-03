import os
import re
import json
import random
import requests
from typing import Dict, List, Optional

# Google GenAI SDK (google-genai). The older google-generativeai package is
# deprecated, and the 2.5 models it defaulted to are closed to new API keys.
try:
    from google import genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

# Cheapest current model by default. Override with GEMINI_MODEL in .env / Railway
# (e.g. "gemini-3.8-flash" for stronger but slower picks).
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"
GEMINI_FALLBACK_MODEL = "gemini-3.1-flash-lite"

# Try importing transformers, but don't fail if missing
try:
    from transformers import pipeline
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

def select_words_with_llm(
    candidates: List[str],
    count: int,
    query_text: str,
    mode: str = "ollama",
    model_name: str = None,
    api_key: str = None,
    rarity: str = "off",
    sources: Optional[Dict[str, str]] = None,
) -> List[str]:
    """
    Selects the best words from a list of candidates using an LLM.

    Args:
        candidates: List of candidate words.
        count: Number of words to select.
        query_text: The original query text (context).
        mode: "ollama", "huggingface", or "gemini".
        model_name: Model name override (defaults depend on mode).
        api_key: API key for Gemini.
        rarity: The poet's rarity setting, so the LLM's taste follows it.
        sources: word -> corpus it came from, so picks can span corpora.

    Returns:
        List of selected words.
    """

    if not candidates:
        return []

    # If we have fewer candidates than requested, just return them all
    if len(candidates) <= count:
        return candidates

    prompt = _build_prompt(candidates, count, query_text, rarity, sources)

    selected_words = []

    try:
        if mode == "ollama":
            # Default to llama3 for Ollama
            model = model_name or "gemma3:4b"
            selected_words = _call_ollama(prompt, model)
        elif mode == "huggingface":
            # Default to a small instruction-following model for HF
            # gpt2 is too dumb. TinyLlama is better.
            model = model_name or "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            selected_words = _call_huggingface(prompt, model)
        elif mode == "gemini":
            selected_words = _call_gemini(prompt, api_key, model_name)
        else:
            print(f"Unknown LLM mode: {mode}. Falling back to random.")
            selected_words = []
            
    except Exception as e:
        print(f"LLM selection failed ({mode}): {e}")
        selected_words = []

    # Fallback if LLM fails or returns nothing
    if not selected_words:
        print("LLM returned no valid words. Falling back to top candidates.")
        # Since candidates are ordered by stanza priority (relevance), 
        # taking the top ones is a reasonable fallback, or random.
        # User said "instead of random select", but if LLM fails...
        # Let's just take the top ones to respect priority.
        return candidates[:count]
        
    # Ensure we return exactly 'count' words if possible, but LLM might return fewer/more.
    # We should filter to ensure they are actually in the candidate list (hallucination check)
    # and preserve the casing from candidates if possible.
    
    candidate_set = {w.lower(): w for w in candidates}
    valid_words = []
    seen = set()
    
    for w in selected_words:
        # Models sometimes echo the "(corpus)" tag from the candidate list.
        w_lower = re.sub(r"\s*\(.*\)$", "", str(w)).lower().strip()
        if w_lower in candidate_set and w_lower not in seen:
            valid_words.append(candidate_set[w_lower])
            seen.add(w_lower)
            
    # If we need more, fill from candidates
    if len(valid_words) < count:
        for w in candidates:
            w_lower = w.lower()
            if w_lower not in seen:
                valid_words.append(w)
                seen.add(w_lower)
                if len(valid_words) >= count:
                    break
                    
    return valid_words[:count]

def generate_shadow_poem(
    words: List[str],
    context_text: str = None,
    mode: str = "ollama",
    model_name: str = None,
    api_key: str = None
) -> dict:
    """
    Generates a poem using the provided words.
    """
    if not words:
        return {"title": "Silence", "body": "No words collected yet."}
        
    words_str = ", ".join(words)
    
    lang_instruction = ""
    if context_text:
        lang_instruction = f"""
Language Instruction:
Detect the language of the following context text: "{context_text[:200]}...".
WRITE THE POEM IN THAT SAME LANGUAGE.
"""
    else:
        lang_instruction = "Language Instruction: Write the poem in the same language as the provided words."

    prompt = f"""
You are a mystical poet inspired from the style of William Blake and Rainer Maria Rilke, as well as modern abstract poets.
Words provided: {words_str}
{lang_instruction}

Task: Write a short evocative poem using AS MANY of the provided words as possible.
The poem should be a "Shadow Poem" - a reflection of the scattered words, providing ambiguous meaning, but deep resonance.
Output format: Return ONLY a JSON object with "title" and "body" fields.
Example: {{"title": "The Shadow", "body": "The words fell...\\nLike rain..."}}
"""

    try:
        if mode == "ollama":
            model = model_name or "llama3"
            return _call_ollama_dict(prompt, model)
        elif mode == "huggingface":
            model = model_name or "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            return _call_huggingface_dict(prompt, model)
        elif mode == "gemini":
            return _call_gemini_dict(prompt, api_key)
        else:
            return {"title": "Error", "body": "Unknown LLM mode."}
    except Exception as e:
        print(f"Shadow poem generation failed: {e}")
        return {"title": "Error", "body": "Could not generate poem."}

# How the selection should lean for each rarity setting in the panel. Before,
# the prompt always demanded rare words, which fought "Prefer common words".
_RARITY_TASTE = {
    "off": "Mix vivid, concrete words with a few rare or unusual ones.",
    "prefer_rare": "Favour rare, unusual, precise words over everyday ones.",
    "only_rare": "Favour rare, unusual, precise words over everyday ones.",
    "prefer_common": "Favour plain, everyday words that still carry a strong image.",
}

# Gemini structured output: the reply is always {"words": [...]}.
_WORDS_SCHEMA = {
    "type": "object",
    "properties": {"words": {"type": "array", "items": {"type": "string"}}},
    "required": ["words"],
}


def _build_prompt(
    candidates: List[str],
    count: int,
    query_text: str,
    rarity: str = "off",
    sources: Optional[Dict[str, str]] = None,
) -> str:
    sources = sources or {}
    listed = "\n".join(
        f"- {w} ({sources[w]})" if sources.get(w) else f"- {w}" for w in candidates
    )
    several_corpora = len({s for s in sources.values() if s}) > 1
    taste = _RARITY_TASTE.get((rarity or "off").lower(), _RARITY_TASTE["off"])

    return f"""
You choose words for RainWords, a poetry instrument. A poet has just written the line below.
The app found passages in its text corpora that sit close to the line in meaning, and pulled
these candidate words from them. Your picks fall as rain around the poet's cursor to feed the next line.

Poet's line: "{query_text}"

Candidates (word, then the corpus it came from):
{listed}

Pick exactly {count} words from the candidates.
- Choose words that resonate with the line's images or feeling and also open a new direction:
  an unexpected angle, a concrete image, a strong verb.
- {taste}
- Keep the picks varied. Never pick two forms of one word (dream, dreams, dreaming) and mix parts of speech.{'''
- The candidates come from several corpora. Draw from more than one of them.''' if several_corpora else ''}
- Skip fragments and broken tokens (qu, l', d', ment), proper nouns, and function words.
- Copy each word exactly as it appears in the list, without the corpus name.

Return ONLY a JSON object: {{"words": ["word1", "word2"]}}
"""

def _call_ollama(prompt: str, model: str) -> List[str]:
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json" # Ollama supports json format enforcement
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        text = data.get("response", "")
        return _parse_json_list(text)
    except Exception as e:
        print(f"Ollama error: {e}")
        raise

def _call_huggingface(prompt: str, model_name: str) -> List[str]:
    if not HAS_TRANSFORMERS:
        raise ImportError("transformers library not installed.")
    
    # This is a bit tricky for "simplest setup" because downloading models takes time/space.
    # We'll use a text-generation pipeline.
    # Warning: This runs locally and can be slow/heavy.
    
    generator = pipeline("text-generation", model=model_name)
    # We might need to adjust max_length etc.
    result = generator(prompt, max_new_tokens=100, num_return_sequences=1)
    text = result[0]['generated_text']
    
    # Extract the JSON part from the text (it might repeat the prompt)
    # This is a naive implementation.
    if prompt in text:
        text = text.replace(prompt, "")
        
    return _parse_json_list(text)

_gemini_clients: Dict[str, "genai.Client"] = {}


def _gemini_client(api_key: Optional[str]) -> "genai.Client":
    if not HAS_GEMINI:
        raise ImportError("google-genai library not installed.")

    # Try to get key from argument, then environment (GEMINI_API_KEY or GOOGLE_API_KEY)
    key_to_use = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key_to_use:
        raise ValueError("Gemini API key is missing. Please set GEMINI_API_KEY or GOOGLE_API_KEY in .env.")

    if key_to_use not in _gemini_clients:
        _gemini_clients[key_to_use] = genai.Client(api_key=key_to_use)
    return _gemini_clients[key_to_use]


def _model_unavailable(err: Exception) -> bool:
    msg = str(err).lower()
    return "404" in msg or "not found" in msg or "no longer available" in msg


# Models that answered "unavailable" once; skipped for the rest of the process.
_unavailable_models: set = set()


def _gemini_generate(
    prompt: str,
    api_key: Optional[str],
    model_name: Optional[str] = None,
    schema: Optional[dict] = None,
    thinking_level: Optional[str] = None,
    json_only: bool = False,
) -> str:
    """One Gemini call through the Interactions API; returns the reply text.

    Tries the configured model, then GEMINI_FALLBACK_MODEL if the first one is
    retired or unknown, so a model shutdown degrades instead of failing.
    """
    client = _gemini_client(api_key)
    models = [model_name or os.environ.get("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL]
    if GEMINI_FALLBACK_MODEL not in models:
        models.append(GEMINI_FALLBACK_MODEL)
    models = [m for m in models if m not in _unavailable_models] or models[-1:]

    # store=False: these are one-shot calls, so nothing is kept server-side.
    kwargs = {"input": prompt, "store": False}
    if schema or json_only:
        kwargs["response_format"] = {"type": "text", "mime_type": "application/json"}
        if schema:
            kwargs["response_format"]["schema"] = schema
    if thinking_level:
        kwargs["generation_config"] = {"thinking_level": thinking_level}

    last_err: Optional[Exception] = None
    for model in models:
        try:
            return client.interactions.create(model=model, **kwargs).output_text or ""
        except Exception as e:
            last_err = e
            if not _model_unavailable(e):
                break
            _unavailable_models.add(model)
            print(f"Gemini model {model} unavailable, trying the next one: {e}")
    print(f"Gemini error: {last_err}")
    raise last_err


def _call_gemini(prompt: str, api_key: str, model_name: Optional[str] = None) -> List[str]:
    # Word picking is a quick judgement, so keep thinking minimal for latency.
    text = _gemini_generate(prompt, api_key, model_name, schema=_WORDS_SCHEMA, thinking_level="minimal")
    return _parse_json_list(text)

def _parse_json_list(text: str) -> List[str]:
    # Find the first '[' and last ']'
    start = text.find('[')
    end = text.rfind(']')
    
    if start != -1 and end != -1:
        json_str = text[start:end+1]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
            
    # Fallback: try to split by commas if JSON fails
    # or regex
    import re
    # Look for words in quotes
    matches = re.findall(r'"([^"]+)"', text)
    if matches:
        return matches
        
    return []

# --- Helpers for Shadow Poem (Dict return) ---

def _parse_json_dict(text: str) -> dict:
    start = text.find('{')
    end = text.rfind('}')
    
    if start != -1 and end != -1:
        json_str = text[start:end+1]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    return {"title": "Error", "body": text}

def _call_ollama_dict(prompt: str, model: str) -> dict:
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json"
    }
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        text = data.get("response", "")
        return _parse_json_dict(text)
    except Exception as e:
        print(f"Ollama error: {e}")
        return {"title": "Error", "body": str(e)}

def _call_huggingface_dict(prompt: str, model_name: str) -> dict:
    if not HAS_TRANSFORMERS:
        return {"title": "Error", "body": "Transformers not installed."}
    
    generator = pipeline("text-generation", model=model_name)
    result = generator(prompt, max_new_tokens=200, num_return_sequences=1)
    text = result[0]['generated_text']
    if prompt in text:
        text = text.replace(prompt, "")
    return _parse_json_dict(text)

def _call_gemini_dict(prompt: str, api_key: str, model_name: Optional[str] = None) -> dict:
    try:
        text = _gemini_generate(prompt, api_key, model_name, json_only=True)
        print(f"DEBUG: Gemini Raw Response: {text}")
        return _parse_json_dict(text)
    except Exception as e:
        print(f"DEBUG: Gemini Error: {e}")
        return {"title": "Error", "body": str(e)}

# --- Root Tracer Logic ---
#
# The model traces each content word one stage at a time (thin <- Old English
# þynne <- Proto-Germanic *þunnuz <- PIE *ten-), held to the reference
# etymologies; code does the grouping. Identical ancestors (same language,
# same normalised form) share a `key`, so the browser can merge lineages into
# real trees in which poem words meet at the roots they share. Amphibians are
# then requested against those roots by id, so each one is anchored to two
# nodes of the tree and names the morpheme it takes from each.

# Measured on six test poems (dev/root_tracer): a trace takes 6-13 s on the
# flash-lite model against 26-46 s on flash, so tracing stays on the fast model.
# Amphibians need wider recall (kins came back richer and more often right on
# flash), and at low thinking flash answers in 3-12 s. A second call checks
# every amphibian's parts against their roots (_verify_amphibians).
# Override with GEMINI_ROOTS_MODEL / GEMINI_AMPHIBIANS_MODEL.
DEFAULT_ROOTS_MODEL = "gemini-3.5-flash-lite"
DEFAULT_AMPHIBIANS_MODEL = "gemini-3.8-flash"
AMPHIBIANS_THINKING = "low"

_STEP = {
    "type": "object",
    "properties": {
        "form": {"type": "string"},
        "lang": {"type": "string"},
        "meaning": {"type": "string"},
        "reconstructed": {"type": "boolean"},
    },
    "required": ["form", "lang", "meaning", "reconstructed"],
}
_ROOTS_SCHEMA = {
    "type": "object",
    "properties": {
        "language": {"type": "string"},
        "words": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "word": {"type": "string"},
                "lemma": {"type": "string"},
                "certainty": {"type": "string", "enum": ["established", "probable", "uncertain"]},
                "lineage": {"type": "array", "items": _STEP},
            },
            "required": ["word", "lemma", "certainty", "lineage"],
        }},
        "roots": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "form": {"type": "string"},
                "lang": {"type": "string"},
                "meaning": {"type": "string"},
                "cousins": {"type": "array", "items": {
                    "type": "object",
                    "properties": {"word": {"type": "string"}, "path": {"type": "array", "items": _STEP}},
                    "required": ["word", "path"],
                }},
            },
            "required": ["form", "lang", "meaning", "cousins"],
        }},
    },
    "required": ["language", "words", "roots"],
}
_AMPHIBIANS_SCHEMA = {
    "type": "object",
    "properties": {
        "amphibians": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "word": {"type": "string"},
                "kind": {"type": "string", "enum": ["twin", "kin"]},
                "parts": {"type": "array", "items": {
                    "type": "object",
                    "properties": {
                        "root_id": {"type": "string"},
                        "morpheme": {"type": "string"},
                        "via": {"type": "string"},
                    },
                    "required": ["root_id", "morpheme", "via"],
                }},
                "touches": {"type": "array", "items": {"type": "string"}},
                "literal": {"type": "string"},
                "echo": {"type": "string"},
            },
            "required": ["word", "kind", "parts", "touches", "literal", "echo"],
        }},
    },
    "required": ["amphibians"],
}

_CHECK_SCHEMA = {
    "type": "object",
    "properties": {
        "checks": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "n": {"type": "integer"},
                "parts_ok": {"type": "array", "items": {"type": "boolean"}},
                "note": {"type": "string"},
            },
            "required": ["n", "parts_ok", "note"],
        }},
    },
    "required": ["checks"],
}

# Spelling variants that should not split one ancestor into two nodes. Marks on
# consonants stay: PIE *weǵ- ("be lively") and *weg- ("weave") are two roots.
_FORM_FOLD = str.maketrans({
    "ʰ": "h", "ʷ": "w", "ʲ": "y", "₁": "1", "₂": "2", "₃": "3", "’": "'",
})
_VOWELS = set("aeiouyæœəɛɔ")
_LANG_ALIASES = {
    "pie": "proto-indo-european",
    "indo-european": "proto-indo-european",
    "classical latin": "latin",
    "greek": "ancient greek",
    "classical greek": "ancient greek",
    "anglo-saxon": "old english",
}


def _lang_key(lang: str) -> str:
    k = re.sub(r"\(.*?\)", "", (lang or "").lower()).strip()
    return _LANG_ALIASES.get(k, k)


def _form_key(form: str) -> str:
    import unicodedata
    f = re.sub(r"\(.*?\)", "", (form or "").lower()).translate(_FORM_FOLD)
    # Drop accents and length marks on vowels (é, ō, ḗ), keep them on consonants (ǵ, ḱ).
    out = []
    for ch in unicodedata.normalize("NFD", f):
        if unicodedata.combining(ch) and out and out[-1] in _VOWELS:
            continue
        out.append(ch)
    f = unicodedata.normalize("NFC", "".join(out))
    return re.sub(r"[\s*/]+", "", f).strip("-")


def _node_key(lang: str, form: str) -> str:
    return f"{_lang_key(lang)}|{_form_key(form)}"


def _roots_prompt(text: str) -> str:
    return f"""
You are a historical linguist. Trace the etymology of the content words of this poem.

Poem:
\"\"\"{text}\"\"\"

"language": the poem's language as an ISO code ("en", "fr", ...).

"words": up to 24 content words (nouns, verbs, adjectives, meaningful adverbs). Skip articles,
pronouns, prepositions, conjunctions, auxiliaries and proper names. For each:
- "word" as written in the poem, "lemma" its dictionary form.
- "lineage": its ancestors from the nearest to the deepest, one stage per step. Each step has
  "form", "lang" (full language name), "meaning" (2 to 5 words) and "reconstructed" (true for
  unattested proto-forms). Examples:
    thin   -> Old English "þynne" -> Proto-Germanic "*þunnuz" -> Proto-Indo-European "*ten-"
    racine -> Late Latin "radicina" -> Latin "radix" -> Proto-Indo-European "*wréh₂ds"
    cathedral -> Old French "chaiere" -> Late Latin "cathedra" -> Ancient Greek "kathédra (καθέδρα)"
             -> Proto-Indo-European "*sed-"
  Go as deep as the reference etymologies allow, down to Proto-Indo-European when a PIE root is accepted.
- "certainty": "established" for the standard etymology of the reference dictionaries (OED,
  Etymonline, Wiktionary, TLFi/CNRTL for French), "probable" when it is the usual view but debated,
  "uncertain" when the origin is unknown or disputed.

Accuracy rules:
- Follow the reference etymologies. Never link a word to a root because of sound or meaning resemblance.
- Where the origin is unknown, stop the lineage at the last known stage and mark the word "uncertain".
  Many Latin, Greek and Germanic words have no accepted PIE root (Latin "puer", "silva", Greek
  "thálassa", English "dog"): their lineage ends at the oldest attested or reconstructed form, with no
  PIE step. A short true lineage is worth more than a long invented one.
- For prefixed or compound words, follow the stem that carries the meaning, not the prefix
  (apprendre -> Latin "apprehendere" -> Latin "prehendere" -> Proto-Indo-European "*gʰed-").
- Write each ancestor the same way everywhere it appears, so ancestors shared by several words are
  identical strings. Write PIE roots in LIV notation with an asterisk (*ten-, *deh₂-, *bʰer-, *gʷem-),
  keeping the marks that separate roots (*weǵ- "be lively" and *weg- "weave" are two roots),
  Greek in Latin transcription with the Greek script in parentheses.

"roots": for every ancestor shared by two or more poem words, and for the deepest ancestor of every word,
give "form" and "lang" exactly as written in the lineages, its "meaning", and 2 to 4 "cousins": real
words in the poem's language that descend from it by a known path, preferably through other branches
than the poem word (for *ten-: "tendon" via Latin, "hypotenuse" via Greek). Cousins must not be poem
words or inflections of them. Give each cousin's "path": its ancestors from the nearest up to, but not
including, the root, in the same step format as the lineages (tendon: Medieval Latin "tendo", Latin
"tendere"). Reuse the exact strings of any ancestor that already appears in a lineage.

Return ONLY a JSON object with "language", "words" and "roots".
"""


def _attested(word: str, lang: str) -> bool:
    """True when wordfreq knows the word in the poem's language (filters cousins
    given in another language, and invented ones)."""
    try:
        from wordfreq import zipf_frequency
        return zipf_frequency(word, lang) >= 1.5 or zipf_frequency(word.lower(), lang) >= 1.5
    except Exception:
        return True   # without wordfreq, keep the model's cousin


def _clean_trace(data: dict, text: str) -> dict:
    """Validate the model's trace and give every ancestor a merge key."""
    from spacy.lang.en.stop_words import STOP_WORDS as EN_STOP
    from spacy.lang.fr.stop_words import STOP_WORDS as FR_STOP

    lang = (data.get("language") or "en").lower()[:2]
    stops = FR_STOP if lang == "fr" else EN_STOP
    words, seen = [], set()
    for w in data.get("words") or []:
        lemma = str(w.get("lemma") or w.get("word") or "").strip()
        low = lemma.lower()
        if not lemma or low in stops or low in seen:
            continue
        steps = []
        for s in w.get("lineage") or []:
            form, lg = str(s.get("form") or "").strip(), str(s.get("lang") or "").strip()
            if not form or not lg:
                continue
            key = _node_key(lg, form)
            if steps and steps[-1]["key"] == key:
                continue   # the same stage listed twice
            # Asterisked forms and proto-languages are reconstructed whatever the flag says.
            recon = bool(s.get("reconstructed")) or form.startswith("*") or lg.lower().startswith("proto")
            steps.append({"form": form, "lang": lg, "meaning": str(s.get("meaning") or "").strip(),
                          "reconstructed": recon, "key": key})
        seen.add(low)
        certainty = w.get("certainty") if w.get("certainty") in ("established", "probable", "uncertain") else "probable"
        words.append({"word": str(w.get("word") or lemma), "lemma": lemma,
                      "certainty": certainty, "lineage": steps})

    in_poem = set(re.findall(r"[\w'’-]+", text.lower())) | seen
    roots = []
    for r in data.get("roots") or []:
        form, lg = str(r.get("form") or "").strip(), str(r.get("lang") or "").strip()
        if not form or not lg:
            continue
        cousins = []
        for c in r.get("cousins") or []:
            cw = str(c.get("word") or "").strip()
            if not cw or cw.lower() in in_poem or len(cousins) >= 4:
                continue
            if not _attested(cw, lang):
                continue   # not a word of the poem's language (e.g. Old Norse "bein")
            path = [{"form": str(st.get("form")).strip(), "lang": str(st.get("lang")).strip(),
                     "meaning": str(st.get("meaning") or "").strip(),
                     "reconstructed": bool(st.get("reconstructed")) or str(st.get("form")).strip().startswith("*")
                                      or str(st.get("lang")).lower().startswith("proto"),
                     "key": _node_key(str(st.get("lang")), str(st.get("form")))}
                    for st in (c.get("path") or []) if st.get("form") and st.get("lang")]
            cousins.append({"word": cw, "path": path,
                            "via": ", ".join(f'{st["lang"]} {st["form"]}' for st in path)})
        roots.append({"form": form, "lang": lg, "meaning": str(r.get("meaning") or "").strip(),
                      "key": _node_key(lg, form), "cousins": cousins})
    return {"version": 2, "language": lang, "words": words, "roots": roots}


def _verify_cousins(trace: dict, api_key, model) -> dict:
    """Ask once whether each cousin truly descends from its root; drop those that don't."""
    items = [(r, c) for r in trace["roots"] for c in r["cousins"]]
    if not items:
        return trace
    listed = "\n".join(
        f'{n}. {c["word"]} <- {c["via"] or "(direct)"} <- {r["form"]} ({r["lang"]}, "{r["meaning"]}")'
        for n, (r, c) in enumerate(items, 1))
    lang_name = {"en": "English", "fr": "French"}.get(trace.get("language"), trace.get("language", ""))
    prompt = f"""
You check etymologies for a poetry tool. Each line claims that a word descends from a root through the
path shown. Decide whether the word truly descends from that root, by the standard etymology
(Wiktionary, Etymonline, TLFi/CNRTL). Look-alikes and homonyms do not count; a loose path is fine if the
descent is real. Each word must also be a word of the poem's language ({lang_name}): reject words of
other languages, including Latin, Old French or English forms in a French poem.

{listed}

Return ONLY a JSON object: {{"checks": [{{"n": 1, "parts_ok": [true], "note": "short reason"}}]}}
"""
    data = _parse_json_dict(_gemini_generate(prompt, api_key, model, schema=_CHECK_SCHEMA, thinking_level="low"))
    ok = {c.get("n"): bool((c.get("parts_ok") or [False])[0]) for c in data.get("checks") or []}
    drop = {id(c) for n, (r, c) in enumerate(items, 1) if not ok.get(n, False)}
    for r in trace["roots"]:
        r["cousins"] = [c for c in r["cousins"] if id(c) not in drop]
    return trace


def trace_roots_with_llm(
    text: str,
    depth: str = "deep",
    mode: str = "ollama",
    model_name: str = None,
    api_key: str = None
) -> dict:
    """
    Trace the etymology of the poem's content words.

    Returns {"version": 2, "language", "words": [{word, lemma, certainty,
    lineage: [{form, lang, meaning, reconstructed, key}, ...]}], "roots":
    [{form, lang, meaning, key, cousins: [{word, via}]}]}. Lineages always go
    as deep as known; `depth` is applied when drawing ("standard" hides the
    reconstructed stages), so switching depth needs no new call.
    """
    if not text or len(text.strip()) < 5:
        return {"version": 2, "language": "en", "words": [], "roots": []}

    prompt = _roots_prompt(text)
    try:
        if mode == "gemini":
            model = model_name or os.environ.get("GEMINI_ROOTS_MODEL") or DEFAULT_ROOTS_MODEL
            raw = _gemini_generate(prompt, api_key, model, schema=_ROOTS_SCHEMA, thinking_level="low")
            trace = _clean_trace(_parse_json_dict(raw), text)
            try:
                checker = os.environ.get("GEMINI_AMPHIBIANS_MODEL") or DEFAULT_AMPHIBIANS_MODEL
                trace = _verify_cousins(trace, api_key, checker)
            except Exception as e:
                print(f"Cousin check failed, showing unchecked cousins: {e}")
            return trace
        elif mode == "ollama":
            data = _call_ollama_dict(prompt, model_name or "llama3")
        elif mode == "huggingface":
            data = _call_huggingface_dict(prompt, model_name or "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        else:
            return {"version": 2, "language": "en", "words": [], "roots": []}
        return _clean_trace(data, text)
    except Exception as e:
        print(f"Root trace failed: {e}")
        return {"version": 2, "language": "en", "words": [], "roots": [], "error": str(e)}

# --- Amphibian Logic ---

def _amphibians_prompt(roots: List[dict], context_text: Optional[str], language: str) -> str:
    listed = "\n".join(
        f'- {r["id"]}: {r.get("form")} ({r.get("lang", "")}) "{r.get("meaning", "")}"'
        + (f' -> in the poem: {", ".join(r["words"])}' if r.get("words") else "")
        for r in roots
    )
    poem = f'The poem:\n"""{context_text}"""\n' if context_text else ""
    return f"""
You find "amphibian" words for a poet: real words that live in two of the poem's families at once.
{poem}
The poem's roots (id: form, language, meaning, and the poem words that descend from it):
{listed}

Find 8 to 10 existing words in the poem's language ("{language}"), none of them already in the poem,
of two kinds:
- "twin": the word's two parts descend from two different roots of the list, by any branch.
  Model: English "philosophy" = phil- (Greek philos, "loving") + -sophy (Greek sophia, "wisdom").
  Twins are rare and the most precious: give every true one you find.
- "kin": the word descends from one root of the list, and its literal etymological sense meets the
  image of a poem word that comes from a different root. Model: in a poem whose roots include *sed-
  ("to sit") and whose words include "snow", "sediment" (Latin sedimentum, "a settling", from *sed-)
  meets the snow that settles. Reach into another branch of the family than the poem word's own:
  not a plain derivative of the poem word ("paternal" for "father", "vocal" for "voice").

Look for twins first and systematically: go through the pairs of roots and think of compounds,
prefixed verbs and derivatives that join them, in the poem's language and through Latin and Greek.

Rules:
- Real dictionary words with standard etymologies. No invented compounds. Each part must truly
  descend from its root: look-alikes do not count ("belfry" has nothing to do with "bell", the Latin
  suffix "-aster" nothing to do with "aster", star).
- "parts": the parts that descend from the listed roots (two for a twin, one for a kin), each with its
  "root_id", the "morpheme" as it appears in the word, and "via", its path back to the root
  (e.g. "Latin sedere <- PIE *sed-").
- "touches": the poem words whose images the word meets (for a kin, at least one poem word that
  descends from a different root than its part).
- "literal": the word's literal etymological sense, short and concrete ("a settling", "love of wisdom").
- "echo": one plain sentence on what that literal sense opens in this poem. No hype.
- Prefer vivid or surprising literal senses. List twins first.

Return ONLY a JSON object: {{"amphibians": [...]}}
"""


def _clean_amphibians(data: dict, roots: List[dict], context_text: Optional[str]) -> dict:
    """
    Keep amphibians anchored to two distinct roots of the tree: a twin through
    two etymological parts, a kin through one part plus a poem word (of another
    root) whose image its literal sense meets. `root_ids` are the etymological
    anchors, `meaning_root_ids` the roots reached through a touched poem word.
    """
    ids = {r["id"] for r in roots}
    root_of_word = {w.lower(): r["id"] for r in roots for w in (r.get("words") or [])}
    in_poem = set(re.findall(r"[\w'’-]+", (context_text or "").lower()))
    twins, kins, seen = [], [], set()
    for a in data.get("amphibians") or []:
        word = str(a.get("word") or "").strip()
        if not word or word.lower() in in_poem or word.lower() in seen:
            continue
        parts = [p for p in (a.get("parts") or []) if str(p.get("root_id")) in ids]
        etym = list(dict.fromkeys(str(p["root_id"]) for p in parts))
        touches = [str(t).strip() for t in (a.get("touches") or []) if str(t).strip()]
        via_meaning = [root_of_word[t.lower()] for t in touches
                       if t.lower() in root_of_word and root_of_word[t.lower()] not in etym]
        if len(etym) >= 2:
            kind = "twin"
        elif len(etym) == 1 and via_meaning:
            kind = "kin"
        else:
            continue   # not anchored to two roots
        seen.add(word.lower())
        item = {
            "word": word,
            "kind": kind,
            "root_ids": etym,
            "meaning_root_ids": list(dict.fromkeys(via_meaning))[:2] if kind == "kin" else [],
            "parts": [{"root_id": str(p["root_id"]), "morpheme": str(p.get("morpheme") or ""),
                       "via": str(p.get("via") or "")} for p in parts],
            "touches": touches,
            "literal": str(a.get("literal") or "").strip(),
            "echo": str(a.get("echo") or "").strip(),
        }
        (twins if kind == "twin" else kins).append(item)
    return {"amphibians": (twins + kins)[:8]}


def _check_prompt(items: List[dict], roots: List[dict], language: str = "the poem's language") -> str:
    by_id = {r["id"]: r for r in roots}
    lines = []
    for n, a in enumerate(items, 1):
        parts = "; ".join(
            f'"{p["morpheme"]}" from {by_id[p["root_id"]].get("form")} '
            f'({by_id[p["root_id"]].get("lang", "")}, "{by_id[p["root_id"]].get("meaning", "")}") via {p["via"]}'
            for p in a["parts"] if p["root_id"] in by_id)
        lines.append(f"{n}. {a['word']}: {parts}")
    return f"""
You check etymologies for a poetry tool. For each word below, decide for each listed part whether it
truly descends from the stated root, by the standard etymology (Wiktionary, Etymonline, TLFi/CNRTL).

- Homonyms do not count: Latin "liber" (book, bark) and "liber" (free) are different words.
- A prefix or suffix that only looks like the root does not count (Latin "-aster" is not "aster", star).
- A part counts when it descends from the root by any branch, even when the path given is loose.
- If the word itself is not an attested dictionary word of {language}, mark all its parts false
  (an invented compound like "moonbone" fails even if both parts are real).

{chr(10).join(lines)}

Return ONLY a JSON object: {{"checks": [{{"n": 1, "parts_ok": [true, false], "note": "short reason"}}]}},
with one boolean per listed part, in order.
"""


def _verify_amphibians(items: List[dict], roots: List[dict], api_key, model, language: str = "the poem's language") -> List[dict]:
    """Drop parts that fail an etymology check, then re-apply the anchoring rule:
    a twin keeps two verified parts; a kin keeps its verified part and its meaning link."""
    if not items:
        return items
    data = _parse_json_dict(_gemini_generate(
        _check_prompt(items, roots, language), api_key, model, schema=_CHECK_SCHEMA, thinking_level="low"))
    verdicts = {c.get("n"): c for c in data.get("checks") or []}
    root_of_word = {w.lower(): r["id"] for r in roots for w in (r.get("words") or [])}
    kept = []
    for n, a in enumerate(items, 1):
        v = verdicts.get(n)
        if v is None:
            continue   # unchecked: leave it out rather than show it unverified
        oks = list(v.get("parts_ok") or [])
        parts = [p for i, p in enumerate(a["parts"]) if i < len(oks) and oks[i]]
        etym = list(dict.fromkeys(p["root_id"] for p in parts))
        meaning = [root_of_word[t.lower()] for t in a.get("touches", [])
                   if t.lower() in root_of_word and root_of_word[t.lower()] not in etym]
        if len(etym) >= 2:
            kept.append({**a, "kind": "twin", "parts": parts, "root_ids": etym, "meaning_root_ids": []})
        elif len(etym) == 1 and meaning:
            kept.append({**a, "kind": "kin", "parts": parts, "root_ids": etym,
                         "meaning_root_ids": list(dict.fromkeys(meaning))[:2]})
    return kept


def find_amphibians_with_llm(
    roots_list: List,
    context_text: str = None,
    mode: str = "ollama",
    model_name: str = None,
    api_key: str = None,
    language: str = None,
) -> dict:
    """
    Find words that live in two of the poem's root families. `roots_list`
    holds {"id", "form", "lang", "meaning", "words"} dicts (plain strings
    still work); every amphibian returned is anchored to two distinct ids.
    """
    roots = []
    for i, r in enumerate(roots_list or []):
        if isinstance(r, str):
            r = {"form": r}
        roots.append({**r, "id": str(r.get("id") or f"r{i + 1}")})
    if len(roots) < 2:
        return {"amphibians": []}

    from .text_pipeline import detect_language
    lang = language or (detect_language(context_text) if context_text else "en")
    prompt = _amphibians_prompt(roots, context_text, lang)
    try:
        if mode == "gemini":
            model = model_name or os.environ.get("GEMINI_AMPHIBIANS_MODEL") or DEFAULT_AMPHIBIANS_MODEL
            data = _parse_json_dict(_gemini_generate(
                prompt, api_key, model, schema=_AMPHIBIANS_SCHEMA, thinking_level=AMPHIBIANS_THINKING))
            found = _clean_amphibians(data, roots, context_text)["amphibians"]
            if len(found) < 4:
                # Generation is uneven (1 candidate one call, 9 the next): ask once more.
                again = _parse_json_dict(_gemini_generate(
                    prompt, api_key, model, schema=_AMPHIBIANS_SCHEMA, thinking_level=AMPHIBIANS_THINKING))
                more = _clean_amphibians(again, roots, context_text)["amphibians"]
                have = {a["word"].lower() for a in found}
                found += [a for a in more if a["word"].lower() not in have]
            try:
                lang_name = {"en": "English", "fr": "French"}.get(lang, lang)
                found = _verify_amphibians(found, roots, api_key, model, lang_name)
            except Exception as e:
                print(f"Amphibian check failed, showing unchecked results: {e}")
            twins = [a for a in found if a["kind"] == "twin"]
            return {"amphibians": (twins + [a for a in found if a["kind"] == "kin"])[:8]}
        elif mode == "ollama":
            data = _call_ollama_dict(prompt, model_name or "llama3")
        elif mode == "huggingface":
            data = _call_huggingface_dict(prompt, model_name or "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        else:
            return {"amphibians": []}
        return _clean_amphibians(data, roots, context_text)
    except Exception as e:
        print(f"Amphibian trace failed: {e}")
        return {"amphibians": [], "error": str(e)}
