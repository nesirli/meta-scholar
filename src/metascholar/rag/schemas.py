from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class LLMCallRecord:
    model: str
    prompt: str
    instructions: str
    answer: str
    question: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    response_time: float = 0.0
    cost: float = 0.0
    sources: list[dict[str, Any]] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
