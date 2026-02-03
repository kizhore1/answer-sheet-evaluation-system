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
    nltk.download('punkt_tab')   # Needed for Python 3.13+
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



# ---------- SIMILARITY ----------
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



def find_missing_keywords(student_answer, keywords):
    student_answer = student_answer.lower()
    missing = []

    for keyword in keywords:
        if keyword.lower() not in student_answer:
            missing.append(keyword)

    return missing


# ---------- SCORING ----------
def convert_similarity_to_marks(similarity, keyword_coverage, max_marks):
    if similarity >= 0.75:
        base_marks = 0.8 * max_marks
    elif similarity >= 0.45:
        base_marks = 0.5 * max_marks
    else:
        base_marks = 0.2 * max_marks

    keyword_bonus = keyword_coverage * (0.2 * max_marks)

    final_marks = base_marks + keyword_bonus
    final_marks = min(final_marks, max_marks)

    return round(final_marks, 2)


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
        feedback = "Good answer. You have covered most of the important points."
    elif quality == "Average":
        feedback = "Average answer. Some important concepts are missing."
    else:
        feedback = "Poor answer. Key concepts are missing and the explanation is weak."

    if missing_keywords:
        feedback += "\nMissing keywords: " + ", ".join(missing_keywords)

    return quality, feedback


# ---------- MAIN PROGRAM ----------
def main():
    download_nltk_data()

    # Example inputs
    student_answer = "Machine learning is part of artificial intelligence"
    model_answer = "Artificial intelligence includes machine learning techniques"

    keywords = [
        "artificial intelligence",
        "machine learning",
        "algorithms",
        "data"
    ]

    max_marks = 10

    similarity = calculate_similarity(student_answer, model_answer)
    keyword_coverage = calculate_keyword_coverage(student_answer, keywords)
    missing_keywords = find_missing_keywords(student_answer, keywords)

    final_marks = convert_similarity_to_marks(
        similarity, keyword_coverage, max_marks
    )

    quality, feedback = generate_feedback(
        similarity, keyword_coverage, missing_keywords
    )

    print("\n--- ANSWER EVALUATION REPORT ---")
    print("Student Answer:", student_answer)
    print("Model Answer:", model_answer)
    print("Similarity Score:", round(similarity, 3))
    print("Keyword Coverage:", round(keyword_coverage, 2))
    print("Marks:", final_marks, "/", max_marks)
    print("Answer Quality:", quality)
    print("Feedback:")
    print(feedback)


# ---------- ENTRY POINT ----------
if __name__ == "__main__":
    main()
