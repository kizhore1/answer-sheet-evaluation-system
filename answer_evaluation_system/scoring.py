from functools import lru_cache

from .config import (
    DEFAULT_KEYWORD_MIN_LEN,
    DEFAULT_KEYWORD_COUNT,
    DEFAULT_KEYWORD_DIVERSITY,
    DEFAULT_KEYWORD_SEMANTIC_THRESHOLD,
    DEFAULT_MAX_MARKS,
)
from .nlp import preprocess_text, expand_with_wordnet
from .similarity import get_embedding_model


@lru_cache(maxsize=1)
def _keybert_model():
    from keybert import KeyBERT  # type: ignore

    return KeyBERT()


def extract_keywords(
    model_answer: str,
    min_len: int = DEFAULT_KEYWORD_MIN_LEN,
    top_n: int = DEFAULT_KEYWORD_COUNT,
    diversity: float = DEFAULT_KEYWORD_DIVERSITY,
) -> list[str]:
    try:
        model = _keybert_model()
        keywords = model.extract_keywords(
            model_answer,
            top_n=top_n,
            use_mmr=True,
            diversity=diversity,
        )
        extracted = [kw for kw, _ in keywords]
    except Exception:
        processed, tokens = preprocess_text(model_answer)
        unique = {t for t in tokens if len(t) >= min_len}
        extracted = []
        for token in processed.split():
            if token in unique and token not in extracted:
                extracted.append(token)

    cleaned = []
    for kw in extracted:
        processed_kw, tokens = preprocess_text(kw)
        if any(len(t) >= min_len for t in tokens) and kw not in cleaned:
            cleaned.append(kw if processed_kw else kw)
    return cleaned


def _semantic_keyword_matches(student_answer: str, keywords: list[str]) -> set[str]:
    if not keywords:
        return set()

    try:
        if __import__("os").getenv("AES_DISABLE_EMBEDDINGS", "").lower() in {"1", "true", "yes"}:
            return set()

        try:
            threshold = float(
                __import__("os").getenv(
                    "AES_KEYWORD_SEMANTIC_THRESHOLD",
                    DEFAULT_KEYWORD_SEMANTIC_THRESHOLD,
                )
            )
        except ValueError:
            threshold = DEFAULT_KEYWORD_SEMANTIC_THRESHOLD

        from sentence_transformers import util  # type: ignore
        model = get_embedding_model()
        embeddings = model.encode([student_answer] + keywords, normalize_embeddings=True)
        student_emb = embeddings[0]
        keyword_embs = embeddings[1:]
        scores = util.cos_sim(student_emb, keyword_embs)[0]
        return {
            keywords[i]
            for i, score in enumerate(scores)
            if float(score) >= threshold
        }
    except Exception:
        return set()


def calculate_keyword_coverage(student_answer: str, keywords: list[str]) -> float:
    if not keywords:
        return 0.0

    _, student_tokens = preprocess_text(student_answer)
    token_set = set(student_tokens)
    expanded = expand_with_wordnet(student_tokens)

    matched_keywords = set()
    for keyword in keywords:
        key_processed, key_tokens = preprocess_text(keyword)
        key_tokens = key_processed.split() if key_processed else key_tokens

        if all(part in token_set or part in expanded for part in key_tokens):
            matched_keywords.add(keyword)

    matched_keywords |= _semantic_keyword_matches(student_answer, keywords)
    return len(matched_keywords) / len(keywords)


def find_missing_keywords(student_answer: str, keywords: list[str]) -> list[str]:
    if not keywords:
        return []

    _, student_tokens = preprocess_text(student_answer)
    token_set = set(student_tokens)
    expanded = expand_with_wordnet(student_tokens)

    matched_keywords = set()
    for keyword in keywords:
        key_processed, key_tokens = preprocess_text(keyword)
        key_tokens = key_processed.split() if key_processed else key_tokens

        if all(part in token_set or part in expanded for part in key_tokens):
            matched_keywords.add(keyword)

    matched_keywords |= _semantic_keyword_matches(student_answer, keywords)
    return [k for k in keywords if k not in matched_keywords]


def convert_similarity_to_marks(
    similarity_score: float,
    keyword_coverage: float,
    max_marks: int = DEFAULT_MAX_MARKS,
) -> float:
    if similarity_score >= 0.75:
        base_marks = 0.8 * max_marks
    elif similarity_score >= 0.45:
        base_marks = 0.5 * max_marks
    else:
        base_marks = 0.2 * max_marks

    keyword_bonus = keyword_coverage * (0.2 * max_marks)
    final_marks = min(base_marks + keyword_bonus, max_marks)
    return round(final_marks, 2)
