from __future__ import annotations

import re
from dataclasses import dataclass

import nltk
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
from sklearn.datasets import fetch_20newsgroups

from pso_tfidf.config import DataConfig


@dataclass
class CorpusBundle:
    texts: list[str]
    labels: list[int]
    target_names: list[str]


def _ensure_nltk_stopwords() -> None:
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)


def preprocess_20newsgroups(config: DataConfig) -> CorpusBundle:
    """Load 20 Newsgroups, clean, and return space-joined token strings."""
    _ensure_nltk_stopwords()
    newsgroups = fetch_20newsgroups(
        subset=config.subset,
        remove=("headers", "footers", "quotes"),
    )
    stop_words = set(stopwords.words("english"))
    stop_words.update(
        {"from", "subject", "re", "edu", "use", "to", "for", "in", "over", "on"}
    )

    processed: list[str] = []
    for doc in newsgroups.data:
        doc = re.sub(r"[,\\.!?]", "", doc).lower()
        tokens = simple_preprocess(doc, deacc=True)
        tokens = [t for t in tokens if t not in stop_words]
        processed.append(" ".join(tokens))

    n = min(config.max_docs, len(processed))
    return CorpusBundle(
        texts=processed[:n],
        labels=list(newsgroups.target[:n]),
        target_names=list(newsgroups.target_names),
    )
