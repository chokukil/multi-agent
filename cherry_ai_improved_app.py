import streamlit as st
try:
    from modules.utils.runtime_diag import attach_diagnostics, maybe_reset_from_query
    from modules.ui.layout_manager import LayoutManager
    _has_layout=True
except Exception:
    from modules.utils.runtime_diag import attach_diagnostics, maybe_reset_from_query
    _has_layout=False
    from modules.ui import emergency_uploader

def main():
    maybe_reset_from_query()
    st.set_page_config(layout="wide")
    st.title("Cherry AI (Improved)")
    if _has_layout:
        try:
            LayoutManager().render()
        except Exception as e:
            st.warning(f"레이아웃 오류: {e}")
            emergency_uploader.render()
    else:
        emergency_uploader.render()
    attach_diagnostics()

if __name__ == "__main__":
    main()