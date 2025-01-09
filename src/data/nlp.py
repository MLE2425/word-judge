import logging
from transformers import AutoTokenizer
import nltk
import string

logger = logging.getLogger(__name__)

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

try:
    stopwords: set[str] = set(nltk.corpus.stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    stopwords: set[str] = set(nltk.corpus.stopwords.words("english"))

punctuation: set[str] = set(string.punctuation)
punctuation.add("’")

try:
    lemmatizer = nltk.stem.WordNetLemmatizer()
except OSError:
    nltk.download("wordnet")
    lemmatizer = nltk.stem.WordNetLemmatizer()


def tokenize(text: str, remove_stopwords: bool, lemmatize: bool) -> list[str]:
    tokens: list[str] = tokenizer.tokenize(text, add_special_tokens=False)

    if remove_stopwords:
        tokens = [token for token in tokens if token not in stopwords]

    if lemmatize:
        tokens = [lemmatizer.lemmatize(token) for token in tokens]

    tokens = [token for token in tokens if token not in punctuation]

    chunk_size: int = 510  # 512 - 2

    chunks: list[str] = [
        tokens[i : i + chunk_size] for i in range(0, len(tokens), chunk_size)
    ]

    chunks = [[tokenizer.cls_token] + chunk + [tokenizer.sep_token] for chunk in chunks]

    if len(chunks) == 1:
        return chunks[0]
    return chunks


def encode(chunks: list[list[str]]) -> list[int]:
    encoded_chunks: list[list[int]] = [
        tokenizer.convert_tokens_to_ids(chunk) for chunk in chunks
    ]
    if len(encoded_chunks) == 1:
        return encoded_chunks[0]
    return encoded_chunks
