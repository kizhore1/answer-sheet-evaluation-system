from .config import DEFAULT_THRESHOLDS, Thresholds


def classify_answer(
    similarity: float,
    keyword_coverage: float,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
) -> str:
    if similarity >= thresholds.good_similarity and keyword_coverage >= thresholds.good_keyword:
        return "Good"
    if similarity >= thresholds.avg_similarity and keyword_coverage >= thresholds.avg_keyword:
        return "Average"
    return "Poor"


def generate_feedback(
    similarity: float,
    keyword_coverage: float,
    missing_keywords: list[str],
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
) -> tuple[str, str]:
    quality = classify_answer(similarity, keyword_coverage, thresholds)

    if quality == "Good":
        feedback = "Good answer. You covered most of the important points."
    elif quality == "Average":
        feedback = "Average answer. Some important points are missing."
    else:
        feedback = "Poor answer. Key concepts are missing."

    if missing_keywords:
        feedback += "\n\nMissing keywords:\n- " + "\n- ".join(missing_keywords)

    return quality, feedback
