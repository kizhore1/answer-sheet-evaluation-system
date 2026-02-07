import os
import streamlit as st

from .config import (
    DEFAULT_MAX_MARKS,
    DEFAULT_SIMILARITY_WEIGHTS,
    DEFAULT_THRESHOLDS,
    Thresholds,
    DEFAULT_KEYWORD_COUNT,
    DEFAULT_KEYWORD_DIVERSITY,
    DEFAULT_KEYWORD_SEMANTIC_THRESHOLD,
    DEFAULT_CROSS_ENCODER_LOW,
    DEFAULT_CROSS_ENCODER_HIGH,
)
from .nlp import download_nltk_data
from .similarity import combined_similarity
from .scoring import (
    extract_keywords,
    calculate_keyword_coverage,
    find_missing_keywords,
    convert_similarity_to_marks,
)
from .feedback import generate_feedback


def run() -> None:
    st.set_page_config(page_title="Answer Evaluation System", layout="centered")
    download_nltk_data()

    st.title("Answer Evaluation System")
    st.write("Compare student answers with model answers using NLP.")

    with st.sidebar:
        st.header("Configuration")
        max_marks = st.number_input(
            "Max Marks",
            min_value=1,
            max_value=100,
            value=DEFAULT_MAX_MARKS,
            step=1,
        )
        tfidf_weight = st.slider(
            "TF-IDF Weight",
            min_value=0.0,
            max_value=1.0,
            value=DEFAULT_SIMILARITY_WEIGHTS.tfidf,
            step=0.05,
        )
        semantic_weight = st.slider(
            "Semantic Weight",
            min_value=0.0,
            max_value=1.0,
            value=DEFAULT_SIMILARITY_WEIGHTS.semantic,
            step=0.05,
        )
        if tfidf_weight == 0 and semantic_weight == 0:
            st.warning("At least one similarity weight must be > 0. Using TF-IDF only.")
            tfidf_weight = 1.0
            semantic_weight = 0.0

        st.subheader("Quality Thresholds")
        good_similarity = st.slider(
            "Good Similarity",
            min_value=0.0,
            max_value=1.0,
            value=DEFAULT_THRESHOLDS.good_similarity,
            step=0.05,
        )
        good_keyword = st.slider(
            "Good Keyword Coverage",
            min_value=0.0,
            max_value=1.0,
            value=DEFAULT_THRESHOLDS.good_keyword,
            step=0.05,
        )
        avg_similarity = st.slider(
            "Average Similarity",
            min_value=0.0,
            max_value=1.0,
            value=DEFAULT_THRESHOLDS.avg_similarity,
            step=0.05,
        )
        avg_keyword = st.slider(
            "Average Keyword Coverage",
            min_value=0.0,
            max_value=1.0,
            value=DEFAULT_THRESHOLDS.avg_keyword,
            step=0.05,
        )

        st.subheader("Keyword Extraction")
        keyword_count = st.slider(
            "Keyword Count",
            min_value=3,
            max_value=20,
            value=DEFAULT_KEYWORD_COUNT,
            step=1,
        )
        keyword_diversity = st.slider(
            "Keyword Diversity (MMR)",
            min_value=0.0,
            max_value=1.0,
            value=DEFAULT_KEYWORD_DIVERSITY,
            step=0.05,
        )
        keyword_semantic_threshold = st.slider(
            "Keyword Semantic Threshold",
            min_value=0.0,
            max_value=1.0,
            value=DEFAULT_KEYWORD_SEMANTIC_THRESHOLD,
            step=0.05,
        )
        os.environ["AES_KEYWORD_SEMANTIC_THRESHOLD"] = str(keyword_semantic_threshold)

        st.subheader("Cross-Encoder")
        cross_low = st.slider(
            "Cross-Encoder Low",
            min_value=0.0,
            max_value=1.0,
            value=DEFAULT_CROSS_ENCODER_LOW,
            step=0.05,
        )
        cross_high = st.slider(
            "Cross-Encoder High",
            min_value=0.0,
            max_value=1.0,
            value=DEFAULT_CROSS_ENCODER_HIGH,
            step=0.05,
        )
        if cross_low > cross_high:
            st.warning("Cross-encoder low should be <= high. Using defaults.")
            cross_low = DEFAULT_CROSS_ENCODER_LOW
            cross_high = DEFAULT_CROSS_ENCODER_HIGH
        os.environ["AES_CROSS_ENCODER_LOW"] = str(cross_low)
        os.environ["AES_CROSS_ENCODER_HIGH"] = str(cross_high)

        disable_embeddings = st.checkbox(
            "Disable Embeddings",
            value=os.getenv("AES_DISABLE_EMBEDDINGS") in {"1", "true", "yes"},
        )
        if disable_embeddings:
            os.environ["AES_DISABLE_EMBEDDINGS"] = "1"
        else:
            os.environ.pop("AES_DISABLE_EMBEDDINGS", None)

        disable_cross = st.checkbox(
            "Disable Cross-Encoder",
            value=os.getenv("AES_DISABLE_CROSS_ENCODER") in {"1", "true", "yes"},
        )
        if disable_cross:
            os.environ["AES_DISABLE_CROSS_ENCODER"] = "1"
        else:
            os.environ.pop("AES_DISABLE_CROSS_ENCODER", None)

    model_answer = st.text_area("Model Answer", height=150)
    student_answer = st.text_area("Student Answer", height=150)

    if st.button("Evaluate"):
        if not model_answer or not student_answer:
            st.warning("Please enter both answers.")
            return

        keywords = extract_keywords(
            model_answer,
            top_n=keyword_count,
            diversity=keyword_diversity,
        )

        tfidf_score, semantic_score, cross_score, similarity = combined_similarity(
            student_answer,
            model_answer,
            weights=type(DEFAULT_SIMILARITY_WEIGHTS)(
                tfidf=tfidf_weight,
                semantic=semantic_weight,
            ),
        )
        keyword_coverage = calculate_keyword_coverage(student_answer, keywords)
        missing_keywords = find_missing_keywords(student_answer, keywords)

        marks = convert_similarity_to_marks(similarity, keyword_coverage, max_marks)
        quality, feedback = generate_feedback(
            similarity,
            keyword_coverage,
            missing_keywords,
            thresholds=Thresholds(
                good_similarity=good_similarity,
                good_keyword=good_keyword,
                avg_similarity=avg_similarity,
                avg_keyword=avg_keyword,
            ),
        )

        st.subheader("Evaluation Result")
        st.write(f"Similarity Score: {similarity:.3f}")
        st.write(f"TF-IDF Score: {tfidf_score:.3f}")
        if semantic_score is not None:
            st.write(f"Semantic Score: {semantic_score:.3f}")
        else:
            st.caption("Semantic model not available. Using TF-IDF only.")
        if cross_score is not None:
            st.write(f"Cross-Encoder Score: {cross_score:.3f}")
        st.write(f"Marks: {marks} / {max_marks}")
        st.write(f"Answer Quality: {quality}")

        st.subheader("Feedback")
        st.write(feedback)


if __name__ == "__main__":
    run()
