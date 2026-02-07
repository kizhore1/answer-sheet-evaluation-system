# Answer Evaluation System

AI-based automated answer sheet evaluation using NLP. It compares a student answer with a model answer and scores similarity even when wording differs but meaning is similar.

## Project Structure

- `answer_evaluation_system/` - Core package
- `app.py` - Streamlit entry point
- `examples/` - Demo scripts
- `tests/` - Pytest tests

## Quick Setup (Windows PowerShell)

1. Check Python:

```powershell
python --version
```

2. Create a virtual environment and install dependencies:

```powershell
cd answer-sheet-evaluation-system
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet')"
```

3. Run tests:

```powershell
.\.venv\Scripts\python -m pytest
```

4. Run the Streamlit app:

```powershell
.\.venv\Scripts\python -m streamlit run app.py
```

## Notes

- If PowerShell blocks scripts, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` before activating.
- Semantic similarity uses `sentence-transformers` if installed. First use may download a model.
- Keyword extraction uses `keybert` (with MMR diversity) when available.
- A cross-encoder can be used for \"gray zone\" semantic scores to improve accuracy.
- To disable embeddings, set `AES_DISABLE_EMBEDDINGS=1` in your environment.
- To disable the cross-encoder, set `AES_DISABLE_CROSS_ENCODER=1` in your environment.
- You can tune thresholds via env vars: `AES_CROSS_ENCODER_LOW`, `AES_CROSS_ENCODER_HIGH`, `AES_KEYWORD_SEMANTIC_THRESHOLD`.

## Simple Explanation (Similarity + Keywords)

Similarity:
- TF-IDF checks word overlap for important terms.
- Semantic model checks meaning even if words differ.
- Cross-encoder is a stronger meaning check used only when the score is uncertain.

Keywords:
- KeyBERT extracts important keywords/phrases from the model answer.
- Matching accepts exact or semantic matches, so synonyms can count.
