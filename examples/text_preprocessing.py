from answer_evaluation_system.nlp import download_nltk_data, preprocess_text


def main() -> None:
    download_nltk_data()

    input_text = "The cats are running faster than the dogs!"
    print("Original Text:")
    print(input_text)

    processed_text, _ = preprocess_text(input_text)

    print("\nPreprocessed Output:")
    print(processed_text)


if __name__ == "__main__":
    main()
