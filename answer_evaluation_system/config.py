from dataclasses import dataclass

SYNONYM_MAP = {
    "technology": "model",
    "delivers": "deliver",
    "delivering": "deliver",
    "provides": "provide",
    "offers": "provide",
    "machines": "system",
    "systems": "system",
    "computers": "system",
    "decision making": "decision",
    "make decisions": "decision",
    "problem solving": "problem solve",
    "solve problems": "problem solve",
    "cost effective": "cost efficiency",
    "cost-effective": "cost efficiency",
    "resource sharing": "shared resources",
}


@dataclass(frozen=True)
class SimilarityWeights:
    tfidf: float = 0.5
    semantic: float = 0.5


@dataclass(frozen=True)
class Thresholds:
    good_similarity: float = 0.70
    good_keyword: float = 0.70
    avg_similarity: float = 0.40
    avg_keyword: float = 0.40


DEFAULT_MAX_MARKS = 10
DEFAULT_KEYWORD_MIN_LEN = 3
DEFAULT_KEYWORD_COUNT = 10
DEFAULT_KEYWORD_DIVERSITY = 0.5
DEFAULT_KEYWORD_SEMANTIC_THRESHOLD = 0.55
DEFAULT_TFIDF_NGRAMS = (1, 2)
DEFAULT_EMBEDDING_MODEL = "all-mpnet-base-v2"
DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/stsb-roberta-base"
DEFAULT_CROSS_ENCODER_LOW = 0.55
DEFAULT_CROSS_ENCODER_HIGH = 0.80
DEFAULT_SIMILARITY_WEIGHTS = SimilarityWeights()
DEFAULT_THRESHOLDS = Thresholds()
