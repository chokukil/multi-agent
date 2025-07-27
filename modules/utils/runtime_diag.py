import os, sys, streamlit as st, platform, pandas as pd
def attach_diagnostics():
    st.markdown('<div data-testid="diagnostics-panel"></div>', unsafe_allow_html=True)
    with st.expander("🩺 Diagnostics", expanded=False):
        try:
            app_path = os.path.abspath(sys.modules["__main__"].__file__)
        except Exception:
            app_path = "(unknown)"
        st.write("**ACTIVE_APP**:", app_path)
        st.write("**Python**:", platform.python_version(), " | **Streamlit**:", st.__version__, " | **Pandas**:", pd.__version__)
        st.write("**uploaded_datasets**:", list(st.session_state.get("uploaded_datasets", {}).keys()))
        st.write("**selected_datasets**:", st.session_state.get("selected_datasets", []))
def maybe_reset_from_query():
    qp = st.query_params
    if qp.get("reset", ["0"])[0] in ("1","true","yes"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.success("🔄 Session reset by query param (?reset=1).")