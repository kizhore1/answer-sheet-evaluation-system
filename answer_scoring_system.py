import string
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from answer_evaluation_system.app import SYNONYM_MAP


# ---------- DOWNLOAD REQUIRED NLTK DATA ----------
def download_nltk_data():
    nltk.download('punkt')
    nltk.download('punkt_tab')   # REQUIRED for Python 3.13+
    nltk.download('stopwords')
    nltk.download('wordnet')


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



# ---------- TF-IDF + COSINE SIMILARITY ----------
def calculate_similarity(student_answer, model_answer):
    student_processed = preprocess_text(student_answer)
    model_processed = preprocess_text(model_answer)

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(
        [student_processed, model_processed]
    )

    similarity = cosine_similarity(
        tfidf_matrix[0], tfidf_matrix[1]
    )[0][0]

    return similarity


# ---------- KEYWORD COVERAGE ----------
def calculate_keyword_coverage(student_answer, keywords):
    student_answer = preprocess_text(student_answer)
    matched = 0

    for keyword in keywords:
        keyword_norm = preprocess_text(keyword)
        key_parts = keyword_norm.split()

        if all(part in student_answer for part in key_parts):
            matched += 1

    return matched / len(keywords)



# ---------- SCORING LOGIC ----------
def convert_similarity_to_marks(similarity_score, keyword_coverage, max_marks):
    if similarity_score >= 0.75:
        base_marks = 0.8 * max_marks
    elif similarity_score >= 0.45:
        base_marks = 0.5 * max_marks
    else:
        base_marks = 0.2 * max_marks

    keyword_bonus = keyword_coverage * (0.2 * max_marks)

    final_marks = base_marks + keyword_bonus
    final_marks = min(final_marks, max_marks)

    return round(final_marks, 2)


# ---------- MAIN PROGRAM ----------
def main():
    download_nltk_data()

    # Example answers
    student_answer = "Machine learning is part of artificial intelligence"
    model_answer = "Artificial intelligence includes machine learning techniques"

    # Important keywords from model answer
    keywords = [
        "artificial intelligence",
        "machine learning",
        "algorithms",
        "data"
    ]

    max_marks = 10

    similarity = calculate_similarity(student_answer, model_answer)
    keyword_coverage = calculate_keyword_coverage(student_answer, keywords)
    final_marks = convert_similarity_to_marks(
        similarity, keyword_coverage, max_marks
    )

    print("\n--- ANSWER EVALUATION RESULT ---")
    print("Student Answer:", student_answer)
    print("Model Answer:", model_answer)
    print("Similarity Score:", round(similarity, 3))
    print("Keyword Coverage:", round(keyword_coverage, 2))
    print("Final Marks:", final_marks, "/", max_marks)


# ---------- ENTRY POINT ----------
if __name__ == "__main__":
    main()
