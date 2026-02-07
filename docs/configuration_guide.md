# Configuration Guide

## Main Flow (Summary)
1. Preprocess model + student answers (lowercase, remove punctuation, lemmatize, normalize synonyms).
2. Similarity
   - TF-IDF similarity (word overlap).
   - Semantic similarity (meaning).
   - Optional cross-encoder in the "gray zone" to refine meaning.
3. Keywords
   - Extract keywords from model (KeyBERT).
   - Check student coverage (exact + semantic matching).
4. Score & Feedback
   - Combine similarity with keyword coverage to produce marks and feedback.

## Slider Effects (Lower = More Forgiving)

Max Marks:
Lower -> final score scale decreases (e.g., max 10 becomes 5).

TF-IDF Weight:
Lower -> TF-IDF matters less in total similarity. If reduced a lot, meaning dominates.

Semantic Weight:
Lower -> meaning similarity matters less. If reduced a lot, word overlap dominates.

Good Similarity:
Lower -> easier to be labeled Good.

Good Keyword Coverage:
Lower -> easier to be Good even with missing key points.

Average Similarity:
Lower -> easier to be labeled Average.

Average Keyword Coverage:
Lower -> easier to be Average with fewer keywords.

Keyword Count:
Lower -> fewer model keywords required, easier for students.

Keyword Diversity (MMR):
Lower -> keywords are more similar to each other (narrow focus).
Higher -> keywords are more diverse (broader coverage).

Keyword Semantic Threshold:
Lower -> easier to count a keyword as matched (more forgiving).
Higher -> stricter semantic match.

Cross-Encoder Low / High:
Defines the "gray zone" where the cross-encoder runs.
Wider range -> cross-encoder triggers more often (slower, more accurate).
Narrow range -> cross-encoder triggers less often.
