import os
from functools import lru_cache
from typing import Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .config import (
    DEFAULT_TFIDF_NGRAMS,
    DEFAULT_SIMILARITY_WEIGHTS,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_CROSS_ENCODER_MODEL,
    DEFAULT_CROSS_ENCODER_LOW,
    DEFAULT_CROSS_ENCODER_HIGH,
    SimilarityWeights,
)
from .nlp import preprocess_text


@lru_cache(maxsize=1)
def _vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(ngram_range=DEFAULT_TFIDF_NGRAMS)


def tfidf_similarity(student_answer: str, model_answer: str) -> float:
    student_processed, _ = preprocess_text(student_answer)
    model_processed, _ = preprocess_text(model_answer)

    vectorizer = _vectorizer()
    tfidf = vectorizer.fit_transform([student_processed, model_processed])
    return float(cosine_similarity(tfidf[0], tfidf[1])[0][0])


@lru_cache(maxsize=1)
def _embedding_model():
    from sentence_transformers import SentenceTransformer  # type: ignore

    model_name = os.getenv("AES_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    return SentenceTransformer(model_name)


def semantic_similarity(student_answer: str, model_answer: str) -> Optional[float]:
    if os.getenv("AES_DISABLE_EMBEDDINGS", "").lower() in {"1", "true", "yes"}:
        return None

    try:
        from sentence_transformers import util  # type: ignore
        model = _embedding_model()
        embeddings = model.encode([student_answer, model_answer], normalize_embeddings=True)
        score = util.cos_sim(embeddings[0], embeddings[1])[0][0]
        return float(score)
    except Exception:
        # If the model isn't available or download fails, fall back to TF-IDF only.
        return None


@lru_cache(maxsize=1)
def _cross_encoder_model():
    from sentence_transformers import CrossEncoder  # type: ignore

    model_name = os.getenv("AES_CROSS_ENCODER_MODEL", DEFAULT_CROSS_ENCODER_MODEL)
    return CrossEncoder(model_name)


def cross_encoder_similarity(student_answer: str, model_answer: str) -> Optional[float]:
    if os.getenv("AES_DISABLE_CROSS_ENCODER", "").lower() in {"1", "true", "yes"}:
        return None


def get_embedding_model():
    return _embedding_model()

    try:
        model = _cross_encoder_model()
        score = model.predict([(student_answer, model_answer)])[0]
        # Cross-encoder models usually output a similarity score in [0, 1].
        return float(score)
    except Exception:
        return None


def combined_similarity(
    student_answer: str,
    model_answer: str,
    weights: SimilarityWeights = DEFAULT_SIMILARITY_WEIGHTS,
) -> tuple[float, float | None, float | None, float]:
    tfidf_score = tfidf_similarity(student_answer, model_answer)
    semantic_score = semantic_similarity(student_answer, model_answer)
    cross_score: float | None = None

    if semantic_score is not None:
        try:
            low = float(os.getenv("AES_CROSS_ENCODER_LOW", DEFAULT_CROSS_ENCODER_LOW))
            high = float(os.getenv("AES_CROSS_ENCODER_HIGH", DEFAULT_CROSS_ENCODER_HIGH))
        except ValueError:
            low = DEFAULT_CROSS_ENCODER_LOW
            high = DEFAULT_CROSS_ENCODER_HIGH

        if low <= semantic_score <= high:
            cross_score = cross_encoder_similarity(student_answer, model_answer)

    semantic_final = cross_score if cross_score is not None else semantic_score

    if semantic_final is None:
        return tfidf_score, None, None, tfidf_score

    total = weights.tfidf + weights.semantic
    combined = (weights.tfidf * tfidf_score + weights.semantic * semantic_final) / total
    return tfidf_score, semantic_score, cross_score, float(combined)
