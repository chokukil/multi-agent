import hashlib, time, io, pandas as pd, streamlit as st
SUPPORTED = ['csv','xlsx','xls','json','parquet','pkl']
def _h(f):
    pos=f.tell(); f.seek(0); h=hashlib.sha256(f.read()).hexdigest(); f.seek(pos); return h
def render():
    st.markdown('<div data-testid="file-upload-section"></div>', unsafe_allow_html=True)
    files = st.file_uploader("Drag and drop your data files here", type=SUPPORTED, accept_multiple_files=True, key="__emuploader__")
    st.session_state.setdefault("uploaded_datasets", {})
    st.session_state.setdefault("uploaded_hashes", set())
    st.session_state.setdefault("selected_datasets", [])
    if not files: return
    processed=0
    for f in files:
        he=_h(f)
        if he in st.session_state["uploaded_hashes"]: continue
        name=f.name; suf=(name.split(".")[-1] or "").lower()
        f.seek(0)
        if   suf=="csv":   df=pd.read_csv(f)
        elif suf in ("xlsx","xls"): df=pd.read_excel(f)
        elif suf=="json": df=pd.read_json(f)
        elif suf=="parquet": df=pd.read_parquet(f)
        elif suf=="pkl":  df=pd.read_pickle(f)
        else: st.warning(f"Unsupported: {suf}"); continue
        st.session_state["uploaded_datasets"][name]=df
        st.session_state["uploaded_hashes"].add(he)
        if name not in st.session_state["selected_datasets"]:
            st.session_state["selected_datasets"].append(name)
        processed+=1
    if processed:
        st.session_state["last_upload_ts"]=time.time()
        st.markdown('<div data-testid="upload-complete"></div>', unsafe_allow_html=True)
    # 미리보기 패널
    if st.session_state["uploaded_datasets"]:
        st.markdown("### 📊 업로드된 데이터셋")
        for name, df in st.session_state["uploaded_datasets"].items():
            with st.container(border=True):
                st.write(f"**{name}** · 행 {len(df):,} · 열 {df.shape[1]:,}")
                with st.expander("📘 데이터 미리보기"):
                    st.dataframe(df.head(10), use_container_width=True)