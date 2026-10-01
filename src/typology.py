"""Interruption typology classifier.

Two implementations behind one interface:

* ``RuleTypologyClassifier``  keyword and duration heuristics. No training, no
  downloads, and a surprisingly strong baseline. Start here.
* ``LearnedTypologyClassifier``  TF-IDF plus logistic regression over the
  partial ASR hypothesis and simple acoustic-proxy features. Trained by
  ``src/train.py``.

The learned model is deliberately small. The paper's claim is about the
recovery policy, not about the classifier, so the classifier should be the
cheapest thing that works. Swap in a fusion model (acoustic + LLM, in the style
of Wang et al. 2024) only once the rest of the pipeline is stable, and always
report the oracle-label condition alongside it.
"""

from __future__ import annotations

import re
from typing import Any

from .interruptions import EVENT_TYPES

_BACKCHANNEL = re.compile(
    r"^\s*(mhm+|mm+|uh[- ]?huh|right|okay|ok|yeah|yep|sure|i see)\s*[.!?]?\s*$",
    re.I,
)
_STOP = re.compile(r"\b(wait|hold on|hang on|stop|no no|shut up|quiet)\b", re.I)
_CORRECTION = re.compile(
    r"\b(no,|not\b|actually|sorry,? i meant|i meant|make it|change it to|rather)\b",
    re.I,
)
_SWITCH = re.compile(
    r"\b(forget that|instead|never mind|different|something else|new (request|search))\b",
    re.I,
)
_RESUME = re.compile(
    r"\b(where were we|go on|carry on|continue|you were saying|sorry)\b", re.I
)


class RuleTypologyClassifier:
    """Keyword heuristics, ordered most specific first."""

    name = "rules"

    def predict(self, utterance: str, features: dict[str, Any] | None = None) -> str:
        text = utterance.strip()
        features = features or {}

        if _BACKCHANNEL.match(text):
            return "backchannel"
        # A topic switch usually also contains correction markers ("actually"),
        # so it has to be tested before self-correction.
        if _SWITCH.search(text):
            return "topic_switch"
        if _CORRECTION.search(text):
            return "self_correction"
        if _STOP.search(text):
            return "barge_in_stop"
        if _RESUME.search(text) or features.get("silence_before_ms", 0) > 1500:
            return "post_pause_resume"
        # Short overlapping speech with no content words is almost always a
        # backchannel; anything longer is the user taking the floor.
        return "backchannel" if len(text.split()) <= 2 else "barge_in_stop"


class LearnedTypologyClassifier:
    """TF-IDF + logistic regression over text and acoustic-proxy features."""

    name = "learned"

    def __init__(self, pipeline: Any) -> None:
        self.pipeline = pipeline

    @staticmethod
    def build() -> Any:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline

        return Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 3), sublinear_tf=True,
                                      min_df=1, analyzer="word")),
            ("clf", LogisticRegression(max_iter=2000, C=4.0,
                                       class_weight="balanced")),
        ])

    @classmethod
    def load(cls, path: str) -> "LearnedTypologyClassifier":
        import joblib

        return cls(joblib.load(path))

    def predict(self, utterance: str, features: dict[str, Any] | None = None) -> str:
        label = self.pipeline.predict([utterance])[0]
        return label if label in EVENT_TYPES else "barge_in_stop"


def get_classifier(cfg: dict):
    which = cfg["restate"].get("typology_model", "rules")
    if which == "rules":
        return RuleTypologyClassifier()
    if which == "learned":
        return LearnedTypologyClassifier.load(cfg["restate"]["typology_checkpoint"])
    raise ValueError(f"unknown typology_model: {which!r}")
