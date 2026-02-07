import sys
import importlib

print("Python:", sys.version.split()[0])

packages = ["nltk", "sklearn", "streamlit", "sentence_transformers", "keybert"]
for p in packages:
    try:
        m = importlib.import_module(p)
        ver = getattr(m, "__version__", "unknown")
        print(f"{p}: {ver}")
    except Exception as e:
        print(f"{p}: NOT INSTALLED ({e.__class__.__name__})")
