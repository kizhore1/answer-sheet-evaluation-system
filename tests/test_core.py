import os

os.environ["AES_DISABLE_EMBEDDINGS"] = "1"

from answer_evaluation_system.nlp import preprocess_text
from answer_evaluation_system.similarity import combined_similarity
from answer_evaluation_system.scoring import (
    extract_keywords,
    calculate_keyword_coverage,
    find_missing_keywords,
)
from answer_evaluation_system.feedback import classify_answer


def test_preprocess_text_basic():
    text = "The cats are running faster than the dogs!"
    processed, tokens = preprocess_text(text)
    assert "cat" in processed or "cats" in tokens
    assert "the" not in tokens


def test_similarity_range():
    student = "Machine learning is part of artificial intelligence"
    model = "Artificial intelligence includes machine learning techniques"
    _, _, _, similarity = combined_similarity(student, model)
    assert 0.0 <= similarity <= 1.0


def test_keyword_coverage_and_missing():
    model = "Artificial intelligence includes machine learning techniques"
    student = "Machine learning is part of artificial intelligence"

    keywords = extract_keywords(model)
    coverage = calculate_keyword_coverage(student, keywords)
    missing = find_missing_keywords(student, keywords)

    assert 0.0 <= coverage <= 1.0
    assert isinstance(missing, list)


def test_feedback_classification():
    assert classify_answer(0.8, 0.8) == "Good"
    assert classify_answer(0.5, 0.5) == "Average"
    assert classify_answer(0.2, 0.2) == "Poor"
