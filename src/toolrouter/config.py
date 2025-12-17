from dataclasses import dataclass
import os

@dataclass(frozen=True)
class QdrantConfig:
    url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    api_key: str | None = os.getenv("QDRANT_API_KEY")
