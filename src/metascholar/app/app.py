import streamlit as st
from metascholar.config import settings

from metascholar.rag.rag_init import assistant


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


if check_auth():
    _, _, right = st.columns([4, 1, 1])
    right.button("Logout", on_click=lambda: st.session_state.update(authenticated=False))

    st.set_page_config(
        page_title="MetaScholar",
        page_icon="🦠"
    )
    st.title("MetaScholar")
    st.markdown("""
    RAG-based question answering app over the metagenomics literature
    """)

    query = st.text_input("Enter your question:")

    st.caption(
        "Try: *What computational pipelines are used for metagenomics analysis?*, "
        "*How does diet affect the gut microbiome?*, "
        "*What tools are used for metagenomic binning?*"
    )

    if st.button("Ask"):
        with st.spinner("Processing..."):
            result = assistant.query(query)
            st.success("Completed!")
            st.write(result.answer)
            if result.sources:
                st.divider()
                st.caption("References")
                for i, src in enumerate(result.sources, 1):
                    st.caption(
                        f"[{i}] {src['title']} "
                        f"({src.get('year', '?')}, {src.get('journal', '?')}) — "
                        f"PMID {src['pmid']}"
                    )
