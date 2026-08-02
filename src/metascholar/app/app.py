import streamlit as st
from metascholar.config import settings


def check_auth():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        return True

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if username and password:
        if username == settings.app_username and password == settings.app_password:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Invalid username or password")
    return False


if check_auth():
    st.sidebar.button(
        "Logout", on_click=lambda: st.session_state.update(authenticated=False)
    )
    st.write("Welcome to the protected app!")
