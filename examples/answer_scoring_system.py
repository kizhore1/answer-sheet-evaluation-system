from answer_evaluation_system.nlp import download_nltk_data
from answer_evaluation_system.similarity import combined_similarity
from answer_evaluation_system.scoring import (
    extract_keywords,
    calculate_keyword_coverage,
    convert_similarity_to_marks,
)


def main() -> None:
    download_nltk_data()

    student_answer = "Machine learning is part of artificial intelligence"
    model_answer = "Artificial intelligence includes machine learning techniques"

    keywords = extract_keywords(model_answer)
    _, _, _, similarity = combined_similarity(student_answer, model_answer)
    keyword_coverage = calculate_keyword_coverage(student_answer, keywords)
    marks = convert_similarity_to_marks(similarity, keyword_coverage, max_marks=10)

    print("\n--- ANSWER EVALUATION RESULT ---")
    print("Student Answer:", student_answer)
    print("Model Answer:", model_answer)
    print("Similarity Score:", round(similarity, 3))
    print("Keyword Coverage:", round(keyword_coverage, 2))
    print("Final Marks:", marks, "/ 10")


if __name__ == "__main__":
    main()
