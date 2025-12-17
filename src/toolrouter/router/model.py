from __future__ import annotations
import os
import pickle
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .actions import Action, DEFAULT_ACTIONS

@dataclass
class RouterModel:
    pipeline: Pipeline
    actions: List[Action]

    def predict_action(self, question: str) -> Action:
        idx = int(self.pipeline.predict([question])[0])
        return self.actions[idx]

def train_router_model(texts: List[str], labels: List[int], actions: List[Action] | None = None) -> RouterModel:
    actions = actions or DEFAULT_ACTIONS
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), max_features=50000, lowercase=True)),
        ("clf", LogisticRegression(max_iter=2000, n_jobs=1, multi_class="auto")),
    ])
    pipe.fit(texts, labels)
    return RouterModel(pipeline=pipe, actions=actions)

def save_router(model: RouterModel, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "router.pkl"), "wb") as f:
        pickle.dump(model, f)

def load_router(dir_path: str) -> RouterModel:
    with open(os.path.join(dir_path, "router.pkl"), "rb") as f:
        return pickle.load(f)
