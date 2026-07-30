"""
Word rarity based on general-language frequency (the `wordfreq` Zipf scale),
independent of which corpora are loaded — so it behaves identically on the
built-in library and on user uploads.

Zipf scale (per wordfreq): ~7 = very common ("the", "le"), ~5 = common,
~3 = uncommon, < 2.5 = rare / precious, 0 = archaic or not in the list.
"""
import random
from functools import lru_cache
from typing import List, Sequence

try:
    from wordfreq import zipf_frequency as _zipf
    HAS_WORDFREQ = True
except Exception:  # pragma: no cover - fallback if the lib is missing
    HAS_WORDFREQ = False

# Thresholds on the Zipf scale (calibrated against FR/EN poetic vocabulary).
RARE_MAX = 3.0      # zipf <= this  -> "rare"
COMMON_MIN = 4.5    # zipf >= this  -> "common"
_NEUTRAL = 3.6      # returned when wordfreq is unavailable (rarity becomes a no-op)


@lru_cache(maxsize=50000)
def zipf(word: str, lang: str) -> float:
    """General-language Zipf frequency of `word` in 'fr' or 'en'."""
    if not HAS_WORDFREQ:
        return _NEUTRAL
    lang = "fr" if (lang or "en").lower().startswith("fr") else "en"
    return _zipf(word.lower(), lang)


def is_rare(word: str, lang: str) -> bool:
    return zipf(word, lang) <= RARE_MAX


def is_common(word: str, lang: str) -> bool:
    return zipf(word, lang) >= COMMON_MIN


def rarity_weight(word: str, lang: str, mode: str) -> float:
    """
    Soft sampling weight (higher => more likely to be picked).
    'prefer_rare' favours low-zipf words; 'prefer_common' favours high-zipf.
    """
    z = zipf(word, lang)
    if mode == "prefer_rare":
        return max(0.05, 8.0 - z) ** 2      # rarer (low z) -> larger weight
    if mode == "prefer_common":
        return max(0.05, z) ** 2            # commoner (high z) -> larger weight
    return 1.0


def weighted_order(items: Sequence[str], weights: Sequence[float]) -> List[str]:
    """
    Weighted random permutation (Efraimidis-Spirakis A-Res): items with higher
    weight tend to come first, but with randomness so results still vary.
    """
    keyed = []
    for it, w in zip(items, weights):
        u = random.random()
        keyed.append((u ** (1.0 / max(w, 1e-6)), it))
    keyed.sort(key=lambda x: x[0], reverse=True)
    return [it for _, it in keyed]
