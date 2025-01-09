import logging
import string
from typing import List

import nltk
import polars as pl
from tqdm import tqdm
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


def tokenize_series(
    series: pl.Series, remove_stopwords: bool = True, lemmatize: bool = True
) -> pl.Series:
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

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

    acc: List[list[str] | list[list[str]]] = [
        tokenize(
            value,
            remove_stopwords,
            lemmatize,
            tokenizer,
            stopwords,
            punctuation,
            lemmatizer,
        )
        for value in tqdm(series, desc="Tokenizing")
        if isinstance(value, str)
    ]
    return pl.Series(acc)


def tokenize(
    text: str,
    remove_stopwords: bool,
    lemmatize: bool,
    tokenizer: AutoTokenizer,
    stopwords: set[str],
    remove_expr: set[str],
    lemmatizer: nltk.stem.WordNetLemmatizer,
) -> list[str] | list[list[str]]:
    tokens: list[str] = tokenizer.tokenize(text, add_special_tokens=False)

    if remove_stopwords:
        tokens = [token for token in tokens if token not in stopwords]

    if lemmatize:
        tokens = [lemmatizer.lemmatize(token) for token in tokens]

    tokens = [token for token in tokens if token not in remove_expr]

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
