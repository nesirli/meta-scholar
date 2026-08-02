import psycopg
from dataclasses import dataclass

from metascholar.config import settings
from metascholar.rag.schemas import LLMCallRecord


def _get_conn():
    return psycopg.connect(
        host=settings.postgres_host,
        dbname=settings.postgres_db,
        user=settings.postgres_user,
        password=settings.postgres_password,
    )


@dataclass
class ConversationRecord:
    question: str
    prompt: str
    answer: str
    model: str
    response_time: float
    cost: float
    timestamp: str


@dataclass
class Stats:
    total: int
    avg_response_time: float
    total_cost: float
    avg_tokens: float


def save_llm_call(record: LLMCallRecord) -> int:
    conn = _get_conn()
    try:
        row = conn.execute(
            """INSERT INTO llm_calls (question, answer, model, instructions, prompt,
               prompt_tokens, completion_tokens, total_tokens, response_time, cost, timestamp)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
               RETURNING id""",
            (record.question, record.answer, record.model, record.instructions,
             record.prompt, record.prompt_tokens, record.completion_tokens,
             record.total_tokens, record.response_time, record.cost, record.timestamp),
        ).fetchone()
        call_id = row[0]
        for rank, src in enumerate(record.sources, 1):
            conn.execute(
                "INSERT INTO llm_call_sources (call_id, pmid, rank) VALUES (%s, %s, %s)",
                (call_id, src["pmid"], rank),
            )
        conn.commit()
        return call_id
    finally:
        conn.close()


def get_conversations(limit: int = 100) -> list[ConversationRecord]:
    conn = _get_conn()
    try:
        rows = conn.execute(
            "SELECT question, prompt, answer, model, response_time, cost, timestamp::text "
            "FROM llm_calls ORDER BY timestamp DESC LIMIT %s",
            (limit,),
        ).fetchall()
        return [ConversationRecord(*r) for r in rows]
    finally:
        conn.close()


def get_stats() -> Stats:
    conn = _get_conn()
    try:
        row = conn.execute(
            "SELECT count(*), coalesce(avg(response_time), 0), "
            "coalesce(sum(cost), 0), coalesce(avg(total_tokens), 0) "
            "FROM llm_calls"
        ).fetchone()
        return Stats(*row)
    finally:
        conn.close()


def get_relevance_stats() -> dict[str, int]:
    conn = _get_conn()
    try:
        rows = conn.execute(
            "SELECT coalesce(relevance, 'UNKNOWN'), count(*) "
            "FROM feedback GROUP BY relevance"
        ).fetchall()
        return {r[0]: r[1] for r in rows}
    finally:
        conn.close()


def get_user_feedback_stats() -> tuple[int, int]:
    conn = _get_conn()
    try:
        up, down = conn.execute(
            "SELECT coalesce(sum(case when source='thumbs_up' then 1 else 0 end), 0), "
            "coalesce(sum(case when source='thumbs_down' then 1 else 0 end), 0) "
            "FROM feedback"
        ).fetchone()
        return up, down
    finally:
        conn.close()
