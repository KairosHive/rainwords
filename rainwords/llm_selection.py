import os
import json
import random
import requests
from typing import List

# Try importing google.generativeai, but don't fail if missing
try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

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
    api_key: str = None
) -> List[str]:
    """
    Selects the best words from a list of candidates using an LLM.
    
    Args:
        candidates: List of candidate words.
        count: Number of words to select.
        query_text: The original query text (context).
        mode: "ollama", "huggingface", or "gemini".
        model_name: Model name for ollama or huggingface.
        api_key: API key for Gemini.
    
    Returns:
        List of selected words.
    """
    
    if not candidates:
        return []
    
    # If we have fewer candidates than requested, just return them all
    if len(candidates) <= count:
        return candidates

    prompt = _build_prompt(candidates, count, query_text)
    
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
            selected_words = _call_gemini(prompt, api_key)
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
        w_lower = w.lower().strip()
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

def _build_prompt(candidates: List[str], count: int, query_text: str) -> str:
    # Join candidates for the prompt
    candidates_str = ", ".join(candidates)
    
    return f"""
You are a poetic assistant. 
Context: "{query_text}"
Candidate words: {candidates_str}

Task: Select exactly {count} words from the candidate list above.
Criteria: Choose the words that are the most semantically distant and rare, with the richest depth of semantic relation. 
Do not choose ANY COMMON OR BORING WORD. Systematically find the most unique and unconventional words.
STRICT FILTER: Do NOT select words that are incomplete, cut off, or look like fragments (e.g., "qu", "l'", "d'", "ment").
Do NOT select proper nouns or names.
Output format: Return ONLY a JSON array of strings. Example: ["word1", "word2"]
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

def _call_gemini(prompt: str, api_key: str) -> List[str]:
    if not HAS_GEMINI:
        raise ImportError("google-generativeai library not installed.")
    
    # Try to get key from argument, then environment (GEMINI_API_KEY or GOOGLE_API_KEY)
    key_to_use = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    
    if not key_to_use:
        raise ValueError("Gemini API key is missing. Please set GEMINI_API_KEY or GOOGLE_API_KEY in .env.")
        
    genai.configure(api_key=key_to_use)
    
    # Use 1.5 Flash for speed
    target_model = "gemini-2.5-flash-lite"
    model = genai.GenerativeModel(target_model) 
    
    try:
        response = model.generate_content(prompt)
        return _parse_json_list(response.text)
    except Exception as e:
        print(f"Gemini error: {e}")
        raise

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

def _call_gemini_dict(prompt: str, api_key: str) -> dict:
    if not HAS_GEMINI:
        return {"title": "Error", "body": "Gemini lib not installed."}
    
    key_to_use = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key_to_use:
        return {"title": "Error", "body": "API Key missing."}
        
    genai.configure(api_key=key_to_use)
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    try:
        response = model.generate_content(prompt)
        print(f"DEBUG: Gemini Raw Response: {response.text}")
        return _parse_json_dict(response.text)
    except Exception as e:
        print(f"DEBUG: Gemini Error: {e}")
        return {"title": "Error", "body": str(e)}

# --- Root Tracer Logic ---

def trace_roots_with_llm(
    text: str,
    depth: str = "deep",
    mode: str = "ollama",
    model_name: str = None,
    api_key: str = None
) -> dict:
    """
    Analyzes the text to find common etymological roots.
    """
    if not text or len(text.strip()) < 5:
        return {"roots": []}

    # Language Instruction for Cousins
    lang_instruction = f"""
Language Instruction:
Detect the language of the text: "{text[:200]}...".
Ensure that the 'cousins' (related words) you suggest are IN THAT SAME LANGUAGE.
"""

    if depth == "deep":
        prompt = f"""
You are an expert etymologist specializing in deep ancestry and Proto-Indo-European (PIE) reconstruction.
Analyze the following text and identify the DEEPEST etymological roots that underpin the words used.

Text: "{text}"
{lang_instruction}

Guidelines:
1. GO DEEP: Do not stop at Latin or Old French. Trace words back to Ancient Greek, Proto-Germanic, or Proto-Indo-European (PIE) whenever possible.
2. PRIORITIZE PIE & Obscure roots: If a Latin/Germanic word comes from a known PIE root (like *bher-, *sta-, *men-), use the PIE root. Be specific about the PIE meaning in the "meaning" field.
3. CONNECTIVITY: Focus on roots that connect multiple words in the text (even if the connection is ancient and not obvious).

Task: Return a JSON object containing a list of "roots".
Structure:
{{
  "roots": [
    {{
      "root": "*root-form",  (Use * for reconstructed roots like PIE)
      "meaning": "meaning of the root",
      "family": "Language Family (e.g. PIE, Ancient Greek, Proto-Germanic)",
      "ancestor": "Optional: If this is a derived root, list its deeper ancestor (e.g. PIE *bher-)",
      "in_poem": ["word1", "word2"],
      "cousins": ["word3", "word4"]
    }}
  ]
}}
Return ONLY valid JSON. Do not include comments in the JSON.
"""
    else:
        # Standard / Shallow Mode
        prompt = f"""
You are an expert etymologist.
Analyze the following text and identify the major etymological roots (Latin, Greek, Germanic) that underpin the words used.

Text: "{text}"
{lang_instruction}

Guidelines:
1. STAY RECOGNIZABLE: Prioritize recognizable roots (e.g., use Latin 'memor' or Greek 'mnemo or mne' for 'memory' rather than the deep PIE '*men-').
2. Focus on roots that connect multiple words in the text, or roots of significant words.
3. PROVIDE ANCESTRY: For Latin/Germanic/Greek roots, identify the deeper PIE or Ancient Greek ancestor if known.

Task: Return a JSON object containing a list of "roots".
Structure:
{{
  "roots": [
    {{
      "root": "root-form",
      "meaning": "meaning of the root",
      "family": "Language Family (e.g. Latin, Greek, Germanic)",
      "ancestor": "The deeper root this comes from (e.g. from Greek '...' or PIE '*...')",
      "in_poem": ["word1", "word2"],
      "cousins": ["word3", "word4"]
    }}
  ]
}}
Return ONLY valid JSON. Do not include comments in the JSON.
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
            return {"roots": []}
    except Exception as e:
        print(f"Root trace failed: {e}")
        return {"roots": []}

# --- Amphibian Logic ---

def find_amphibians_with_llm(
    roots_list: List[str],
    context_text: str = None,
    mode: str = "ollama",
    model_name: str = None,
    api_key: str = None
) -> dict:
    """
    Finds 'amphibian' words that connect two or more roots from the provided list.
    """
    if not roots_list or len(roots_list) < 2:
        return {"amphibians": []}

    roots_str = ", ".join(roots_list)
    
    lang_instruction = "Find EXISTING English words"
    if context_text:
        lang_instruction = f"""
Language Instruction:
Detect the language of the following context text: "{context_text[:200]}...".
Find EXISTING words IN THAT SAME LANGUAGE.
"""
    
    prompt = f"""
You are a creative etymologist and linguist.
I have a list of etymological roots: [{roots_str}].

Task: {lang_instruction} that are "amphibians" — meaning they conceptually or etymologically bridge TWO of these roots.
STRICT CONSTRAINT: Do NOT invent words. Do NOT provide lazy compound words (like "nightangel" or "firewater").
The words must be real, dictionary words that share an etymological ancestry with both roots, OR serve as a strong semantic bridge between them.

Return a JSON object with a list of "amphibians".
Structure:
{{
  "amphibians": [
    {{
      "word": "amphibian_word",
      "root1": "root_A",
      "root2": "root_B",
      "explanation": "Brief reason for the link"
    }}
  ]
}}
Return ONLY valid JSON.
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
            return {"amphibians": []}
    except Exception as e:
        print(f"Amphibian trace failed: {e}")
        return {"amphibians": []}
