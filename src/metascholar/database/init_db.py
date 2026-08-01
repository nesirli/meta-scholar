import os
import psycopg
from datetime import datetime

from metascholar.config import settings

DB_TIMEZONE = datetime.now().astimezone().tzinfo
print(f"Using timezone: {DB_TIMEZONE}")

def get_db_connection():
    return psycopg.connect(
        host=settings.POSTGRES_HOST,
        dbname=settings.POSTGRES_DB,
        user=settings.POSTGRES_USER,
        password=settings.POSTGRES_PASSWORD,
    )

def init_db():
    conn = get_db_connection()
    try:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute("""
                     CREATE TABLE IF NOT EXISTS articles
                     (
                         pmid TEXT PRIMARY KEY,
                         title TEXT,
                         abstract TEXT,
                         year TEXT,
                         journal TEXT,
                         embedding VECTOR(1536)
                     )
                     """)
        conn.execute("""
                     CREATE INDEX IF NOT EXISTS idx_articles_embedding
                     ON articles USING hnsw (embedding vector_cosine_ops)
                     """)
        conn.commit()
    finally:
        conn.close()

def init_conversations():
    conn = get_db_connection()
    try:
        cur.execute("""
                    CREATE TABLE IF NOT EXISTS conversations
                    (
                        id SERIAL PRIMARY KEY,
                        question TEXT NOT NULL,
                        answer TEXT NOT NULL,
                        model TEXT NOT NULL,
                        instructions TEXT NOT NULL,
                        prompt TEXT NOT NULL,
                        prompt_tokens INTEGER NOT NULL,
                        completion_tokens INTEGER NOT NULL,
                        total_tokens INTEGER NOT NULL,
                        response_time FLOAT NOT NULL,
                        cost FLOAT NOT NULL,
                        timestamp TIMESTAMP WITH TIME ZONE NOT NULL
                    )
                    """)
        conn.commit()
    finally:
        conn.close()

def init_feedback():
    conn = get_db_connection()
    try:
        conn.execute("DROP TABLE IF EXISTS feedback")
        conn.execute("""
                    CREATE TABLE IF NOT EXISTS feedback
                    (
                        id SERIAL PRIMARY KEY,
                        conversation_id INTEGER REFERENCES conversations(id),
                        source TEXT NOT NULL,
                        relevance TEXT,
                        explanation TEXT,
                        score INTEGER,
                        timestamp TIMESTAMP WITH TIME ZONE NOT NULL
                    )
                    """)
        conn.commit()
    finally:
        conn.close()

def index_corpus(conn: psycopg.Connection, corpus_path: Path):
    with corpus_path.open() as f:
        for line in f:
            r = json.loads(line)
            emb = embed(r["abstract"])
            conn.execute(
                """INSERT INTO articles (pmid, title, abstract, year, journal, embedding)
                   VALUES (%s, %s, %s, %s, %s, %s)
                   ON CONFLICT (pmid) DO NOTHING""",
                (r["pmid"], r["title"], r["abstract"], r["year"], r["journal"], emb),
            )
    conn.commit()


if __name__ == '__main__':
    init_db()
    init_conversations()
    init_feedback()
    index_corpus(get_db_connection(), settings.CORPUS_PATH)
    print('Database initialized and corpus indexed.')

