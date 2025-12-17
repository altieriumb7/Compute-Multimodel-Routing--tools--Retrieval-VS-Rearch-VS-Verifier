from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

ToolName = Literal["bm25", "dense", "stop"]

@dataclass(frozen=True)
class Action:
    tool: ToolName
    k: int = 0
    def is_stop(self) -> bool:
        return self.tool == "stop"

DEFAULT_ACTIONS = [
    Action("bm25", 5),
    Action("bm25", 20),
    Action("dense", 5),
    Action("dense", 20),
    Action("stop", 0),
]
