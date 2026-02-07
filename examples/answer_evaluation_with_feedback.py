from answer_evaluation_system.nlp import download_nltk_data
from answer_evaluation_system.similarity import combined_similarity
from answer_evaluation_system.scoring import (
    extract_keywords,
    calculate_keyword_coverage,
    find_missing_keywords,
    convert_similarity_to_marks,
)
from answer_evaluation_system.feedback import generate_feedback


def main() -> None:
    download_nltk_data()

    student_answer = "Machine learning is part of artificial intelligence"
    model_answer = "Artificial intelligence includes machine learning techniques"

    keywords = extract_keywords(model_answer)

    _, _, _, similarity = combined_similarity(student_answer, model_answer)
    keyword_coverage = calculate_keyword_coverage(student_answer, keywords)
    missing_keywords = find_missing_keywords(student_answer, keywords)

    marks = convert_similarity_to_marks(similarity, keyword_coverage, max_marks=10)
    quality, feedback = generate_feedback(similarity, keyword_coverage, missing_keywords)

    print("\n--- ANSWER EVALUATION REPORT ---")
    print("Student Answer:", student_answer)
    print("Model Answer:", model_answer)
    print("Similarity Score:", round(similarity, 3))
    print("Keyword Coverage:", round(keyword_coverage, 2))
    print("Marks:", marks, "/ 10")
    print("Answer Quality:", quality)
    print("Feedback:")
    print(feedback)


if __name__ == "__main__":
    main()
