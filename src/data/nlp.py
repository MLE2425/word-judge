import logging
import string
from typing import List

import nltk
import polars as pl
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

logger = logging.getLogger(__name__)


def tokenize_series(
    series: pl.Series, remove_stopwords: bool = True, lemmatize: bool = True
) -> pl.Series:
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

    acc: List[str] = [
        tokenize(
            value,
            remove_stopwords,
            lemmatize,
            tokenizer,
            remove_stopwords and stopwords,
            punctuation,
            lemmatize and lemmatizer,
        ).__str__()
        for value in tqdm(series, desc="Tokenizing")
        if isinstance(value, str)
    ]
    return pl.Series(acc)


def tokenize(
    text: str,
    remove_stopwords: bool,
    lemmatize: bool,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    stopwords: set[str] | bool,
    remove_expr: set[str],
    lemmatizer: nltk.stem.WordNetLemmatizer | bool,
) -> list[str] | list[list[str]]:
    tokens: list[str] = tokenizer.tokenize(text, add_special_tokens=False)

    if remove_stopwords and type(stopwords) is set:
        tokens = [token for token in tokens if token not in stopwords]

    if lemmatize and type(lemmatizer) is nltk.stem.WordNetLemmatizer:
        tokens = [lemmatizer.lemmatize(word=token) for token in tokens]

    tokens = [token for token in tokens if token not in remove_expr]

    chunk_size: int = 512

    chunks: list[list[str]] = [
        tokens[i : i + chunk_size] for i in range(0, len(tokens), chunk_size)
    ]

    del tokens

    return chunks


def encode(chunks: list[list[str]]) -> list[int] | list[list[int]]:
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast = (
        AutoTokenizer.from_pretrained("bert-base-uncased")
    )

    encoded_chunks: list[list[int]] = [
        tokenizer.convert_tokens_to_ids(chunk) for chunk in chunks
    ]

    return encoded_chunks
