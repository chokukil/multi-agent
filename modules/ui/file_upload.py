import io, json, time, pandas as pd, streamlit as st
from modules.utils.ui_singleton import render_once

SUPPORTED = ['csv','xlsx','xls','json','parquet','pkl']

class EnhancedFileUpload:
    def render(self):
        if not render_once("file-upload"):
            return  # prevent duplicate uploader

        st.markdown('<div data-testid="file-upload-section"></div>', unsafe_allow_html=True)
        files = st.file_uploader(
            "Drag and drop your data files here",
            type=SUPPORTED,
            accept_multiple_files=True,
            key="file_uploader"
        )
        if not files:
            return

        # init session stores
        st.session_state.setdefault("uploaded_datasets", {})
        st.session_state.setdefault("selected_datasets", [])

        processed = 0
        for f in files:
            name = f.name
            if name in st.session_state["uploaded_datasets"]:
                continue
            suffix = (name.split(".")[-1] or "").lower()
            try:
                if suffix == "csv":
                    df = pd.read_csv(f)
                elif suffix in ("xlsx","xls"):
                    df = pd.read_excel(f)
                elif suffix == "json":
                    df = pd.read_json(f)
                elif suffix == "parquet":
                    import pyarrow as pa, pyarrow.parquet as pq  # noqa
                    df = pd.read_parquet(f)
                elif suffix == "pkl":
                    df = pd.read_pickle(f)
                else:
                    st.warning(f"Unsupported type: {suffix}")
                    continue
                st.session_state["uploaded_datasets"][name] = df
                if name not in st.session_state["selected_datasets"]:
                    st.session_state["selected_datasets"].append(name)
                processed += 1
            except Exception as e:
                st.error(f"파일 처리 중 오류: {name} — {e}")

        if processed:
            st.session_state["last_upload_ts"] = time.time()
            st.markdown('<div data-testid="upload-complete"></div>', unsafe_allow_html=True)