import streamlit as st

def render_once(key: str) -> bool:
    """Return True if allowed to render now; False if already rendered in this session rerun."""
    flag_key = f"__rendered__{key}"
    if st.session_state.get(flag_key):
        return False
    st.session_state[flag_key] = True
    return True