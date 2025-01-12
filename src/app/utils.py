import logging
import string

import skops.io as sio
import nltk
import sklearn

from transformers import AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from src.cfg import CFG

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def load_model() -> tuple[
    sklearn.base.BaseEstimator, sklearn.feature_extraction.text.TfidfVectorizer
]:
    """
    Load the model and vectorizer from disk

    Returns:
        tuple[sklearn.base.BaseEstimator, sklearn.feature_extraction.text.TfidfVectorizer]: Model and vectorizer
    """
    try:
        logger.info("Loading model")
        with open(
            f"{CFG.project_dir}/output/logistic_regressor.skops", "rb"
        ) as model_file:
            model = sio.load(model_file)
    except Exception:
        logger.error("Failed to load model")
        exit(1)

    try:
        logger.info("Loading vectorizer")
        with open(
            f"{CFG.project_dir}/output/tfidf_vectorizer.skops", "rb"
        ) as vectorizer_file:
            tfidf_vectorizer = sio.load(vectorizer_file)
    except Exception:
        logger.error("Failed to load vectorizer")
        exit(1)

    return model, tfidf_vectorizer


def load_tokenizer() -> tuple[
    PreTrainedTokenizer | PreTrainedTokenizerFast,
    nltk.stem.WordNetLemmatizer,
    set[str],
    set[str],
]:
    """
    Load the tokenizer, lemmatiser, and stopwords

    Returns:
        tuple[PreTrainedTokenizer | PreTrainedTokenizerFast, nltk.stem.WordNetLemmatizer, set[str], set[str]]:\
            Tokenizer, lemmatiser, stopwords, and punctuation
    """
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast = (
        AutoTokenizer.from_pretrained("bert-base-uncased")
    )

    punctuation: set[str] = set(string.punctuation)
    punctuation.add("’")

    try:
        lemmatizer = nltk.stem.WordNetLemmatizer()
    except LookupError:
        nltk.download("wordnet")
        lemmatizer = nltk.stem.WordNetLemmatizer()

    try:
        stopwords = set(nltk.corpus.stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords")
        stopwords = set(nltk.corpus.stopwords.words("english"))

    return tokenizer, lemmatizer, stopwords, punctuation
