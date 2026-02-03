import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from answer_evaluation_system.app import SYNONYM_MAP


# ---------- DOWNLOAD REQUIRED NLTK DATA (RUNS ONCE) ----------
def download_nltk_data():
    nltk.download('punkt')
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


# ---------- SIMILARITY CALCULATION ----------
def calculate_similarity(student_answer, model_answer):
    # Preprocess both texts
    student_processed = preprocess_text(student_answer)
    model_processed = preprocess_text(model_answer)

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(
        [student_processed, model_processed]
    )

    # Cosine Similarity
    similarity = cosine_similarity(
        tfidf_matrix[0], tfidf_matrix[1]
    )[0][0]

    return similarity


# ---------- MAIN FUNCTION ----------
def main():
    download_nltk_data()

    # Example answers
    student_answer = "Machine learning is part of artificial intelligence"
    model_answer = "Artificial intelligence includes machine learning techniques"

    score = calculate_similarity(student_answer, model_answer)

    print("Student Answer:", student_answer)
    print("Model Answer:", model_answer)
    print("\nSimilarity Score:", round(score, 3))


# ---------- ENTRY POINT ----------
if __name__ == "__main__":
    main()
