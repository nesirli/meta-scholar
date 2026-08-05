import time

from pydantic import BaseModel
from typing import Literal
from openai import OpenAI

from metascholar.config import settings
from metascholar.rag.rag_init import assistant


class RelevanceVerdict(BaseModel):
    relevance: Literal["NON_RELEVANT", "PARTLY_RELEVANT", "RELEVANT"]
    explanation: str


JUDGE_INSTRUCTIONS = (
    "You are an expert evaluator for a RAG system. "
    "Analyze the relevance of the generated answer to the given question. "
    "Classify the answer as RELEVANT, PARTLY_RELEVANT, or NON_RELEVANT. "
    "Provide a brief explanation."
)

JUDGE_PROMPT = """Question: {question}

Generated Answer: {answer}"""


def evaluate_relevance(
    question: str,
    answer: str,
    client: OpenAI | None = None,
    max_retries: int = 3,
) -> tuple[str, str]:
    """Score answer relevance with an LLM judge, retrying on transient failures."""
    if client is None:
        client = OpenAI(api_key=settings.openai_api_key)

    prompt = JUDGE_PROMPT.format(question=question, answer=answer)
    messages = [
        {"role": "system", "content": JUDGE_INSTRUCTIONS},
        {"role": "user", "content": prompt},
    ]

    for attempt in range(max_retries):
        try:
            response = client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=messages,
                response_format=RelevanceVerdict,
            )
            result = response.choices[0].message.parsed
            return result.relevance, result.explanation
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)


if __name__ == "__main__":
    question = "What computational pipelines are used for metagenomics analysis?"
    record = assistant.query(question)
    relevance, explanation = evaluate_relevance(question, record.answer)

    print(f"Q: {question}\n")
    print(f"A: {record.answer}\n")
    print(f"Relevance: {relevance}")
    print(f"Explanation: {explanation}")
