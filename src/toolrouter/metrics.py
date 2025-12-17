from __future__ import annotations
import re
from typing import Any, Dict, List

def normalize(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9à-öø-ÿ ]+", "", s)
    return s

def answer_list(ans: Any) -> List[str]:
    if ans is None:
        return []
    if isinstance(ans, str):
        return [ans]
    if isinstance(ans, list):
        return [str(a) for a in ans]
    return [str(ans)]

def retrieval_contains_answer(retrieved_text: str, answer: Any) -> bool:
    t = normalize(retrieved_text)
    for a in answer_list(answer):
        na = normalize(a)
        if na and na in t:
            return True
    return False

def exact_match(pred: str, answer: Any) -> bool:
    np = normalize(pred)
    for a in answer_list(answer):
        if normalize(a) == np:
            return True
    return False
