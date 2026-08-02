import streamlit as st
st.set_page_config(page_title="MetaScholar", page_icon="🦠")

from metascholar.config import settings

from metascholar.rag.rag_init import assistant
from metascholar.rag.judge import evaluate_relevance
from metascholar.app.dashboard import show_dashboard
from metascholar.app.db_query import save_llm_call
from metascholar.app.db_feedback import save_feedback


def check_auth():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        return True

    _, center, _ = st.columns([1, 2, 1])
    with center:
        with st.container(border=True):
            username = st.text_input("Username", placeholder="admin")
            password = st.text_input("Password", type="password", placeholder="••••••••")
            if st.button("Login", use_container_width=True):
                if username == settings.app_username and password == settings.app_password:
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("Invalid username or password")
    return False


def show_chat():
    st.title("MetaScholar")
    st.markdown("RAG-based question answering over the metagenomics literature")

    query = st.text_input("Enter your question:")

    st.caption(
        "Try: *What computational pipelines are used for metagenomics analysis?*, "
        "*How does diet affect the gut microbiome?*, "
        "*What tools are used for metagenomic binning?*"
    )

    if st.button("Ask"):
        with st.spinner("Processing..."):
            result = assistant.query(query)
            conversation_id = save_llm_call(result)
            st.session_state.conversation_id = conversation_id
            st.success("Completed!")
            st.write(result.answer)

            with st.spinner("Evaluating relevance..."):
                relevance, explanation = evaluate_relevance(query, result.answer)
                save_feedback(conversation_id, "judge", relevance=relevance, explanation=explanation)

            if result.sources:
                st.divider()
                st.caption("References")
                for i, src in enumerate(result.sources, 1):
                    st.caption(
                        f"[{i}] {src['title']} "
                        f"({src.get('year', '?')}, {src.get('journal', '?')}) — "
                        f"PMID {src['pmid']}"
                    )

            col1, col2 = st.columns(2)
            with col1:
                if st.button("👍 Helpful"):
                    save_feedback(st.session_state.conversation_id, "user", score=1)
                    st.success("Thanks!")
            with col2:
                if st.button("👎 Not helpful"):
                    save_feedback(st.session_state.conversation_id, "user", score=-1)
                    st.success("Thanks for the feedback!")


if check_auth():
    with st.sidebar:
        if st.button("Logout"):
            st.session_state.authenticated = False
            st.rerun()

    chat_tab, dash_tab = st.tabs(["Chat", "Dashboard"])

    with chat_tab:
        show_chat()
    with dash_tab:
        show_dashboard()
