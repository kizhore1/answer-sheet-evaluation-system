import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from answer_evaluation_system.app import SYNONYM_MAP


# ---------- DOWNLOAD REQUIRED NLTK DATA (RUNS ONLY ONCE) ----------
def download_nltk_data():
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')


# ---------- TEXT PREPROCESSING FUNCTION ----------
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




# ---------- MAIN FUNCTION ----------
def main():
    download_nltk_data()

    # Example input text
    input_text = "The cats are running faster than the dogs!"

    print("Original Text:")
    print(input_text)

    processed_text = preprocess_text(input_text)

    print("\nPreprocessed Output:")
    print(processed_text)


# ---------- PROGRAM ENTRY POINT ----------
if __name__ == "__main__":
    main()
