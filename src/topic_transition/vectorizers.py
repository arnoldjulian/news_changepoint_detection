"""Define vectorizers for text data."""
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from topic_transition.data import preprocess_text


class BaseVectorizer(ABC):
    """Base class for vectorizers."""

    @abstractmethod
    def fit_transform(self, texts: pd.Series | list[str]) -> csr_matrix | np.ndarray:
        """Abstract method for transforming texts."""
        ...


class TFIDFVectorizer(BaseVectorizer):
    """TFIDF vectorizer class."""

    def __init__(self) -> None:
        """Vectorizer creation."""
        self.vectorizer = TfidfVectorizer(stop_words="english")

    def fit_transform(self, texts: pd.Series | list[str]) -> csr_matrix:
        """Fits a TFIDF vectorizer and returns encoded sparse vectors."""
        texts = [preprocess_text(text) for text in texts]
        return self.vectorizer.fit_transform(texts)


class LLMVectorizer(BaseVectorizer):
    """Vectorizer that uses an LLM to get embeddings."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """Creation of SentenceTransformer."""
        self.model = SentenceTransformer(model_name)

    def fit_transform(self, texts: pd.Series | list[str]) -> np.ndarray:
        """Encode texts into embeddings."""
        return self.transform(texts)

    def transform(self, texts: pd.Series | list[str]) -> np.ndarray:
        """Encode texts into embeddings."""
        if isinstance(texts, pd.Series):
            return self.model.encode(texts.to_list())  # type: ignore
        else:
            return self.model.encode(texts)  # type: ignore


def get_vectorizer(vectorizer_type: str) -> BaseVectorizer:
    """Vectorizer factory method."""
    if vectorizer_type == "tfidf":
        return TFIDFVectorizer()
    elif vectorizer_type in ["all-distilroberta-v1", "all-MiniLM-L6-v2"]:
        return LLMVectorizer(vectorizer_type)
    else:
        raise ValueError(f"Invalid vectorizer type: {vectorizer_type}")
