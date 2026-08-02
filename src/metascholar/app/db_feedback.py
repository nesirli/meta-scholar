import psycopg

from metascholar.config import settings


def _get_conn():
    return psycopg.connect(
        host=settings.postgres_host,
        dbname=settings.postgres_db,
        user=settings.postgres_user,
        password=settings.postgres_password,
    )


def save_feedback(
    conversation_id: int,
    source: str,
    relevance: str | None = None,
    explanation: str | None = None,
    score: int | None = None,
) -> None:
    conn = _get_conn()
    try:
        conn.execute(
            """INSERT INTO feedback (llm_call_id, source, relevance, explanation, score, timestamp)
               VALUES (%s, %s, %s, %s, %s, NOW())""",
            (conversation_id, source, relevance, explanation, score),
        )
        conn.commit()
    finally:
        conn.close()
