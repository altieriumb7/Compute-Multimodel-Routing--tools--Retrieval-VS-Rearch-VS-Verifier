from __future__ import annotations
import re
from typing import List

_WORD = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]+", re.UNICODE)

def simple_tokenize(text: str) -> List[str]:
    # Lowercase + keep alnum words; good enough for MVP BM25.
    return [m.group(0).lower() for m in _WORD.finditer(text)]
