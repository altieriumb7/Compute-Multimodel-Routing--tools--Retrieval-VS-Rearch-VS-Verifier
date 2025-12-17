from __future__ import annotations
import orjson
from typing import Iterable, Dict, Any, List

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(orjson.loads(line))
    return rows

def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    with open(path, "wb") as f:
        for r in rows:
            f.write(orjson.dumps(r))
            f.write(b"\n")
