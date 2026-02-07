from answer_evaluation_system.nlp import download_nltk_data
from answer_evaluation_system.similarity import combined_similarity


def main() -> None:
    download_nltk_data()

    student_answer = "Machine learning is part of artificial intelligence"
    model_answer = "Artificial intelligence includes machine learning techniques"

    _, _, _, similarity = combined_similarity(student_answer, model_answer)

    print("Student Answer:", student_answer)
    print("Model Answer:", model_answer)
    print("\nSimilarity Score:", round(similarity, 3))


if __name__ == "__main__":
    main()
