import io, time, hashlib, pandas as pd, streamlit as st
from modules.utils.ui_singleton import render_once
from dataclasses import dataclass
from typing import Dict, Any
import uuid

SUPPORTED = ['csv','xlsx','xls','json','parquet','pkl']

def _hash_filelike(f) -> str:
    pos = f.tell()
    f.seek(0)
    h = hashlib.sha256(f.read()).hexdigest()
    f.seek(pos)
    return h

@dataclass
class QualityIndicators:
    quality_score: float = 100.0
    missing_values: int = 0
    data_types: Dict[str, str] = None
    
    def __post_init__(self):
        if self.data_types is None:
            self.data_types = {}

# Import VisualDataCard from models module
from modules.models import VisualDataCard

class EnhancedFileUpload:
    def render(self):
        # Always render file upload UI - remove singleton restriction
        # if not render_once("file-upload"):
        #     return
        st.markdown('<div data-testid="file-upload-section"></div>', unsafe_allow_html=True)
        files = st.file_uploader(
            "Drag and drop your data files here",
            type=SUPPORTED,
            accept_multiple_files=True,
            key="file_uploader"
        )
        if not files:
            return
        
        # Initialize session state for VisualDataCard objects
        if "uploaded_datasets" not in st.session_state:
            st.session_state.uploaded_datasets = []
        if "uploaded_hashes" not in st.session_state:
            st.session_state.uploaded_hashes = set()

        processed = 0
        existing_names = {card.name for card in st.session_state.uploaded_datasets}
        
        for f in files:
            h = _hash_filelike(f)
            if h in st.session_state["uploaded_hashes"] or f.name in existing_names:
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
                
                # Create VisualDataCard
                memory_usage = f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB"
                quality_indicators = QualityIndicators(
                    quality_score=max(0, 100 - (df.isnull().sum().sum() / df.size * 100)),
                    missing_values=df.isnull().sum().sum(),
                    data_types={col: str(dtype) for col, dtype in df.dtypes.items()}
                )
                
                card = VisualDataCard(
                    id=str(uuid.uuid4()),
                    name=name,
                    file_path=name,  # Store filename as file_path
                    rows=len(df),
                    columns=len(df.columns),
                    preview=df.head(10),
                    data=df,  # Store full dataset
                    quality_indicators=quality_indicators,
                    memory_usage=memory_usage,
                    format=suffix.upper(),
                    metadata={
                        'upload_time': time.strftime('%Y-%m-%d %H:%M:%S'),
                        'column_names': list(df.columns),
                        'column_types': {col: str(dtype) for col, dtype in df.dtypes.items()}
                    }
                )
                
                st.session_state.uploaded_datasets.append(card)
                st.session_state["uploaded_hashes"].add(h)
                processed += 1
                
            except Exception as e:
                st.error(f"파일 처리 중 오류: {name} — {e}")
        
        if processed:
            st.session_state["last_upload_ts"] = time.time()
            st.success(f"✅ {processed}개 파일이 성공적으로 업로드되었습니다!")
            st.markdown('<div data-testid="upload-complete"></div>', unsafe_allow_html=True)