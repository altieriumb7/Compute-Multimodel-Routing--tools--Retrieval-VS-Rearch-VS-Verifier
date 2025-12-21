from __future__ import annotations

from dataclasses import dataclass
import os
import pickle
from typing import List, Sequence

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .actions import Action, DEFAULT_ACTIONS


@dataclass
class RouterModel:
    pipeline: Pipeline
    actions: List[Action]

    def predict_action_idx(self, text: str) -> int:
        return int(self.pipeline.predict([text])[0])

    def predict_action(self, text: str) -> Action:
        idx = self.predict_action_idx(text)
        return self.actions[idx]


def train_router_model(texts: Sequence[str], labels: Sequence[int], actions: Sequence[Action] = DEFAULT_ACTIONS) -> RouterModel:
    pipe = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=200_000)),
            # sklearn >= 1.8: avoid multi_class / n_jobs args; use balanced to fight label skew
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )
    pipe.fit(list(texts), list(labels))
    return RouterModel(pipeline=pipe, actions=list(actions))


def save_router(model: RouterModel, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "router.pkl")
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_router(dir_path: str) -> RouterModel:
    path = os.path.join(dir_path, "router.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)
