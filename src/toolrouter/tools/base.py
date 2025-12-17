from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Protocol

@dataclass
class Doc:
    doc_id: str
    title: str
    text: str
    score: float

@dataclass
class ToolResult:
    docs: List[Doc]
    cost: float
    meta: Dict[str, Any]

class Tool(Protocol):
    name: str
    def retrieve(self, query: str, k: int) -> ToolResult: ...
