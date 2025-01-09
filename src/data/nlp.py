import logging
import string

import nltk
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


def tokenize(
    text: str, remove_stopwords: bool, lemmatize: bool
) -> list[str] | list[list[str]]:
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    punctuation: set[str] = set(string.punctuation)
    punctuation.add("’")

    tokens: list[str] = tokenizer.tokenize(text, add_special_tokens=False)

    if remove_stopwords:
        try:
            stopwords: set[str] = set(nltk.corpus.stopwords.words("english"))
            tokens = [token for token in tokens if token not in stopwords]
        except LookupError:
            nltk.download("stopwords")
            stopwords = set(nltk.corpus.stopwords.words("english"))
            tokens = [token for token in tokens if token not in stopwords]

    if lemmatize:
        try:
            lemmatizer = nltk.stem.WordNetLemmatizer()
            tokens = [lemmatizer.lemmatize(token) for token in tokens]
        except LookupError:
            nltk.download("wordnet")
            lemmatizer = nltk.stem.WordNetLemmatizer()
            tokens = [lemmatizer.lemmatize(token) for token in tokens]

    tokens = [token for token in tokens if token not in punctuation]

    chunk_size: int = 510  # 512 - 2

    chunks: list[list[str]] = [
        tokens[i : i + chunk_size] for i in range(0, len(tokens), chunk_size)
    ]

    chunks = [[tokenizer.cls_token] + chunk + [tokenizer.sep_token] for chunk in chunks]

    if len(chunks) == 1:
        return chunks[0]
    return chunks


def encode(chunks: list[list[str]]) -> list[int] | list[list[int]]:
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    encoded_chunks: list[list[int]] = [
        tokenizer.convert_tokens_to_ids(chunk) for chunk in chunks
    ]
    if len(encoded_chunks) == 1:
        return encoded_chunks[0]
    return encoded_chunks
