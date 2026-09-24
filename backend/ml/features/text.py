"""TF-IDF text feature transformers (fit-on-train).

Concats resume/job text fields and applies TF-IDF.  Vocabulary is learned
only from the training split; ``FittedVectorizer.transform`` can be called
on any split.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from ml.preprocessing.text import normalize


def _safe_val(row: pd.Series, col: str) -> object:
    try:
        val = row[col]
    except KeyError:
        return None
    try:
        if pd.isna(val):
            return None
    except (ValueError, TypeError):
        pass
    return val


def _concat_resume_text(row: pd.Series) -> str:
    parts: list[str] = []
    for col in ("summary", "experience_bullets", "skills"):
        val = _safe_val(row, col)
        if val is not None:
            if isinstance(val, (list, tuple)):
                parts.append(" ".join(str(v) for v in val))
            else:
                parts.append(str(val))
    return " ".join(parts)


def _concat_job_text(row: pd.Series) -> str:
    parts: list[str] = []
    for col in ("description", "responsibilities", "requirements"):
        val = _safe_val(row, col)
        if val is not None:
            if isinstance(val, (list, tuple)):
                parts.append(" ".join(str(v) for v in val))
            else:
                parts.append(str(val))
    return " ".join(parts)


def _normalize_text(text: str) -> str:
    return " ".join(normalize(text).lower().split())


@dataclass(frozen=True)
class FittedVectorizer:
    """A fitted TF-IDF vectorizer that maps pair rows to feature matrices."""

    vectorizer: TfidfVectorizer
    prefix: str

    def transform(self, pairs: pd.DataFrame) -> pd.DataFrame:
        texts = self._build_texts(pairs)
        matrix = self.vectorizer.transform(texts)
        feature_names = self.vectorizer.get_feature_names_out()
        columns = [f"{self.prefix}__{name}" for name in feature_names]
        return pd.DataFrame(
            matrix.toarray(), index=pairs.index, columns=columns, dtype=float
        )

    def _build_texts(self, pairs: pd.DataFrame) -> list[str]:
        if self.prefix == "resume":
            return [
                _normalize_text(_concat_resume_text(row))
                for _, row in pairs.iterrows()
            ]
        return [_normalize_text(_concat_job_text(row)) for _, row in pairs.iterrows()]


@dataclass(frozen=True)
class TextVectorizer:
    """Trainable TF-IDF transformer for resume or job text."""

    prefix: str
    max_features: int | None = None
    min_df: int = 1
    ngram_range: tuple[int, int] = (1, 2)

    def fit(self, pairs_train: pd.DataFrame, y_train: Any = None) -> FittedVectorizer:
        texts = self._build_texts(pairs_train)
        vec = TfidfVectorizer(
            max_features=self.max_features,
            min_df=self.min_df,
            ngram_range=self.ngram_range,
            sublinear_tf=True,
            dtype=np.float64,
        )
        vec.fit(texts)
        return FittedVectorizer(vectorizer=vec, prefix=self.prefix)

    def _build_texts(self, pairs: pd.DataFrame) -> list[str]:
        if self.prefix == "resume":
            return [
                _normalize_text(_concat_resume_text(row))
                for _, row in pairs.iterrows()
            ]
        return [_normalize_text(_concat_job_text(row)) for _, row in pairs.iterrows()]


class ResumeTextVectorizer(TextVectorizer):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(prefix="resume", **kwargs)


class JobTextVectorizer(TextVectorizer):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(prefix="job", **kwargs)
