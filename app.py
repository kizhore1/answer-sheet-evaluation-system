import string
import nltk
import streamlit as st

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SYNONYM_MAP = {
    "technology": "model",
    "delivers": "deliver",
    "delivering": "deliver",
    "provides": "provide",
    "offers": "provide",
    "machines": "system",
    "systems": "system",
    "computers": "system",
    "decision making": "decision",
    "make decisions": "decision",
    "problem solving": "problem solve",
    "solve problems": "problem solve",
    "cost effective": "cost efficiency",
    "cost-effective": "cost efficiency",
    "resource sharing": "shared resources",
}

# ---------- DOWNLOAD NLTK DATA ----------
@st.cache_resource
def download_nltk_data():
    nltk.download("punkt")
    nltk.download("punkt_tab")   # Required for Python 3.13+
    nltk.download("stopwords")
    nltk.download("wordnet")


# ---------- TEXT PREPROCESSING ----------
def normalize_synonyms(text):
    text = text.lower()
    for phrase, normalized in SYNONYM_MAP.items():
        text = text.replace(phrase, normalized)
    return text


def preprocess_text(text):
    text = normalize_synonyms(text)
    text = text.replace("-", " ")
    text = text.translate(str.maketrans("", "", string.punctuation))

    tokens = word_tokenize(text)

    stop_words = set(stopwords.words("english"))
    tokens = [t for t in tokens if t not in stop_words]

    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return " ".join(tokens)



# ---------- SIMILARITY ----------
def calculate_similarity(student_answer, model_answer):
    student_processed = preprocess_text(student_answer)
    model_processed = preprocess_text(model_answer)

    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(
        [student_processed, model_processed]
    )

    return cosine_similarity(tfidf[0], tfidf[1])[0][0]


# ---------- KEYWORD LOGIC ----------
def extract_keywords(model_answer):
    """
    Simple keyword extraction from model answer.
    """
    processed = preprocess_text(model_answer)
    return list(set(processed.split()))


def calculate_keyword_coverage(student_answer, keywords):
    student_answer = student_answer.lower()
    matched = sum(1 for k in keywords if k in student_answer)
    return matched / len(keywords) if keywords else 0


def find_missing_keywords(student_answer, keywords):
    student_answer = student_answer.lower()
    return [k for k in keywords if k not in student_answer]


# ---------- SCORING ----------
def convert_similarity_to_marks(similarity, keyword_coverage, max_marks=10):
    if similarity >= 0.75:
        base = 0.8 * max_marks
    elif similarity >= 0.45:
        base = 0.5 * max_marks
    else:
        base = 0.2 * max_marks

    bonus = keyword_coverage * (0.2 * max_marks)
    return round(min(base + bonus, max_marks), 2)


# ---------- FEEDBACK ----------
def classify_answer(similarity, keyword_coverage):
    if similarity >= 0.7 and keyword_coverage >= 0.7:
        return "Good"
    elif similarity >= 0.4 and keyword_coverage >= 0.4:
        return "Average"
    else:
        return "Poor"


def generate_feedback(similarity, keyword_coverage, missing_keywords):
    quality = classify_answer(similarity, keyword_coverage)

    if quality == "Good":
        feedback = "Good answer. You covered most of the important points."
    elif quality == "Average":
        feedback = "Average answer. Some important points are missing."
    else:
        feedback = "Poor answer. Key concepts are missing."

    if missing_keywords:
        feedback += "\n\nMissing keywords:\n- " + "\n- ".join(missing_keywords)

    return quality, feedback


# ---------- STREAMLIT UI ----------
def main():
    st.set_page_config(page_title="Answer Evaluation System", layout="centered")
    download_nltk_data()

    st.title("📘 Answer Evaluation System")
    st.write("Compare student answers with model answers using NLP.")

    model_answer = st.text_area("Model Answer", height=150)
    student_answer = st.text_area("Student Answer", height=150)

    if st.button("Evaluate"):
        if not model_answer or not student_answer:
            st.warning("Please enter both answers.")
            return

        keywords = extract_keywords(model_answer)

        similarity = calculate_similarity(student_answer, model_answer)
        keyword_coverage = calculate_keyword_coverage(student_answer, keywords)
        missing_keywords = find_missing_keywords(student_answer, keywords)

        marks = convert_similarity_to_marks(similarity, keyword_coverage)
        quality, feedback = generate_feedback(
            similarity, keyword_coverage, missing_keywords
        )

        st.subheader("📊 Evaluation Result")
        st.write(f"**Similarity Score:** {similarity:.3f}")
        st.write(f"**Marks:** {marks} / 10")
        st.write(f"**Answer Quality:** {quality}")
        st.subheader("📝 Feedback")
        st.write(feedback)


if __name__ == "__main__":
    main()

