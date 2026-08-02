from dataclasses import asdict

import pandas as pd
import streamlit as st

from metascholar.app.db_query import (
    get_conversations,
    get_relevance_stats,
    get_stats,
    get_user_feedback_stats,
)


def show_dashboard():
    st.title("Dashboard")

    stats = get_stats()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total conversations", stats.total)
    col2.metric("Avg response time", f"{stats.avg_response_time:.2f}s")
    col3.metric("Total cost", f"${stats.total_cost:.4f}")
    col4.metric("Avg tokens", f"{stats.avg_tokens:.0f}")

    records = get_conversations(limit=100)
    if not records:
        st.info("No conversations yet. Ask a question first!")
        return

    df = pd.DataFrame([asdict(r) for r in records])

    st.subheader("Cost over time")
    st.line_chart(df, x="timestamp", y="cost")

    st.subheader("Response time over time")
    st.line_chart(df, x="timestamp", y="response_time")

    relevance = get_relevance_stats()
    if relevance:
        st.subheader("Judge relevance")
        st.bar_chart(relevance)

    thumbs_up, thumbs_down = get_user_feedback_stats()
    if thumbs_up or thumbs_down:
        st.subheader("User feedback")
        col1, col2 = st.columns(2)
        col1.metric("Thumbs up", int(thumbs_up or 0))
        col2.metric("Thumbs down", int(thumbs_down or 0))

    st.subheader("Recent conversations")
    st.dataframe(
        pd.DataFrame([{
            "Question": r.question[:60],
            "Answer": r.answer[:80],
            "Model": r.model,
            "Time": f"{r.response_time:.1f}s",
            "Cost": f"${r.cost:.4f}",
            "Timestamp": r.timestamp[:19],
        } for r in records[:20]]),
        column_config={
            "Question": st.column_config.TextColumn(width=200),
            "Answer": st.column_config.TextColumn(width=300),
            "Model": st.column_config.TextColumn(width=120),
            "Time": st.column_config.TextColumn(width=70),
            "Cost": st.column_config.TextColumn(width=80),
            "Timestamp": st.column_config.TextColumn(width=140),
        },
        use_container_width=True,
        hide_index=True,
    )
