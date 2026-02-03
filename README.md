# Answer Evaluation System

This is a starter project for a Python-based AI project.

Quick setup (Windows PowerShell):

1. Open PowerShell and run:

   python --version

2. Create a virtual environment and install dependencies:

   cd answer_evaluation_system
   python -m venv .venv
   .\.venv\Scripts\python -m pip install --upgrade pip
   .\.venv\Scripts\python -m pip install -r requirements.txt
   .\.venv\Scripts\python -c "import nltk; nltk.download('punkt')"

3. Run tests:

   .\.venv\Scripts\python test_setup.py

4. Run the Streamlit app:

   .\.venv\Scripts\python -m streamlit run app.py

Notes:
- If PowerShell blocks scripts, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` before activating.
- Ask for help if any step fails.
