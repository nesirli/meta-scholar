from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class LLMCallRecord:
    model: str
    prompt: str
    instructions: str
    answer: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    response_time: float
    cost: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
