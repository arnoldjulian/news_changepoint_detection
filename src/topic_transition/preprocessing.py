"""Tools for preprocessing."""

import spacy

LANGUAGE_MODEL = spacy.load("en_core_web_sm")


def preprocess_text(text: str) -> str:
    """
    Preprocesses text.

    Parameters
    ----------
    text
        Unprocessed article text.
    Returns
    -------
        Preprocessed article text.
    """
    text = text.lower()
    doc = LANGUAGE_MODEL(text)
    tokens = [
        token.lemma_ for token in doc if not token.is_stop and token.is_alpha and not token.is_punct and len(token) > 2
    ]

    processed_text = " ".join(tokens)
    return processed_text
