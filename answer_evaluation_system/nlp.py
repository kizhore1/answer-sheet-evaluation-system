import re
import string
from functools import lru_cache

import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from .config import SYNONYM_MAP


def download_nltk_data() -> None:
    nltk.download("punkt")
    nltk.download("punkt_tab")  # Required for Python 3.13+
    nltk.download("stopwords")
    nltk.download("wordnet")


@lru_cache(maxsize=1)
def _stop_words() -> set[str]:
    return set(stopwords.words("english"))


@lru_cache(maxsize=1)
def _lemmatizer() -> WordNetLemmatizer:
    return WordNetLemmatizer()


def _apply_synonym_map(text: str) -> str:
    # Replace longer phrases first to avoid partial collisions
    for phrase in sorted(SYNONYM_MAP, key=len, reverse=True):
        normalized = SYNONYM_MAP[phrase]
        # Word-boundary replacement to avoid substring artifacts
        pattern = r"\b" + re.escape(phrase) + r"\b"
        text = re.sub(pattern, normalized, text)
    return text


def preprocess_text(text: str) -> tuple[str, list[str]]:
    """
    Normalize, tokenize, remove stopwords, lemmatize.
    Returns (processed_text, tokens).
    """
    text = text.lower()
    text = _apply_synonym_map(text)
    text = text.replace("-", " ")
    text = text.translate(str.maketrans("", "", string.punctuation))

    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in _stop_words()]
    tokens = [_lemmatizer().lemmatize(t) for t in tokens]

    return " ".join(tokens), tokens


def expand_with_wordnet(tokens: list[str]) -> set[str]:
    """
    Expand tokens with WordNet synonyms for light semantic matching.
    This is intentionally conservative to avoid noise.
    """
    expanded = set(tokens)
    for token in tokens:
        if not token.isalpha() or len(token) < 3:
            continue
        for syn in wordnet.synsets(token):
            for lemma in syn.lemmas():
                name = lemma.name().replace("_", " ").lower()
                if name.isalpha() and len(name) >= 3:
                    expanded.add(name)
    return expanded
