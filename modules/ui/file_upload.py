import io, time, hashlib, pandas as pd, streamlit as st
from modules.utils.ui_singleton import render_once

SUPPORTED = ['csv','xlsx','xls','json','parquet','pkl']

def _hash_filelike(f) -> str:
    pos = f.tell()
    f.seek(0)
    h = hashlib.sha256(f.read()).hexdigest()
    f.seek(pos)
    return h

class EnhancedFileUpload:
    def render(self):
        if not render_once("file-upload"):
            return
        st.markdown('<div data-testid="file-upload-section"></div>', unsafe_allow_html=True)
        files = st.file_uploader(
            "Drag and drop your data files here",
            type=SUPPORTED,
            accept_multiple_files=True,
            key="file_uploader"
        )
        if not files:
            return
        st.session_state.setdefault("uploaded_datasets", {})
        st.session_state.setdefault("uploaded_hashes", set())
        st.session_state.setdefault("selected_datasets", [])

        processed = 0
        for f in files:
            h = _hash_filelike(f)
            if h in st.session_state["uploaded_hashes"]:
                continue
            name = f.name
            suffix = (name.split(".")[-1] or "").lower()
            try:
                if suffix == "csv":
                    f.seek(0); df = pd.read_csv(f)
                elif suffix in ("xlsx","xls"):
                    f.seek(0); df = pd.read_excel(f)
                elif suffix == "json":
                    f.seek(0); df = pd.read_json(f)
                elif suffix == "parquet":
                    f.seek(0); df = pd.read_parquet(f)
                elif suffix == "pkl":
                    f.seek(0); df = pd.read_pickle(f)
                else:
                    st.warning(f"Unsupported type: {suffix}"); continue
                st.session_state["uploaded_datasets"][name] = df
                st.session_state["uploaded_hashes"].add(h)
                if name not in st.session_state["selected_datasets"]:
                    st.session_state["selected_datasets"].append(name)
                processed += 1
            except Exception as e:
                st.error(f"파일 처리 중 오류: {name} — {e}")
        if processed:
            st.session_state["last_upload_ts"] = time.time()
            st.markdown('<div data-testid="upload-complete"></div>', unsafe_allow_html=True)