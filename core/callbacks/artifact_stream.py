import streamlit as st
import logging
from typing import Dict, Any, List, Set
from core.artifact_system import artifact_manager
from langchain_core.messages import AIMessage, ToolMessage
from core.utils.streamlit_context import safe_rerun, has_streamlit_context

class ArtifactStreamCallback:
    """
    Monitors the stream for new or updated artifacts and triggers a UI rerun 
    to display them in real-time.
    """
    def __init__(self):
        self._known_artifact_ids: Set[str] = set()
        self._last_check = 0
        self._check_interval = 2  # 최소 2초 간격으로 체크
        self.initialize_state()
        logging.info(f"ArtifactStreamCallback initialized with {len(self._known_artifact_ids)} known artifacts.")

    def initialize_state(self):
        """Initializes or re-initializes the known artifacts from the session."""
        try:
            self._known_artifact_ids = self._get_current_artifact_ids()
        except Exception as e:
            logging.warning(f"Failed to initialize artifact state: {e}")
            self._known_artifact_ids = set()

    def _get_current_artifact_ids(self) -> Set[str]:
        """Retrieves a set of current artifact IDs from the artifact_manager for the current session."""
        if not has_streamlit_context():
            return set()
        
        try:
            import streamlit as st
            if "session_id" not in st.session_state:
                return set()
            # list_artifacts should return a list of dictionaries, each with an 'id'
            artifacts = artifact_manager.list_artifacts(session_id=st.session_state.session_id)
            return {artifact['id'] for artifact in artifacts}
        except Exception as e:
            logging.error(f"Error fetching artifact IDs: {e}")
            return set()

    def __call__(self, msg: Dict[str, Any], **kwargs):
        """
        This method is called for each new message in the graph stream.
        It checks if new artifacts have been created or updated and reruns the UI if so.
        """
        try:
            import time
            
            # 너무 자주 체크하지 않도록 간격 제한
            current_time = time.time()
            if current_time - self._last_check < self._check_interval:
                return
            
            self._last_check = current_time
            
            current_artifact_ids = self._get_current_artifact_ids()
            
            # Check for any difference between current and known artifacts
            if current_artifact_ids != self._known_artifact_ids:
                new_ids = current_artifact_ids - self._known_artifact_ids
                if new_ids:
                    logging.info(f"🎨 ArtifactStreamCallback: Detected {len(new_ids)} new artifact(s): {new_ids}")
                    
                    # 세션 상태에 새 아티팩트 알림 저장 (강제 새로고침 대신)
                    if has_streamlit_context():
                        try:
                            import streamlit as st
                            if hasattr(st.session_state, 'new_artifacts'):
                                st.session_state.new_artifacts.extend(new_ids)
                            else:
                                st.session_state.new_artifacts = list(new_ids)
                        except Exception as e:
                            logging.warning(f"Failed to update session state: {e}")
                
                # Update the known list
                self._known_artifact_ids = current_artifact_ids
                
                # UI 새로고침은 Final_Responder 완료 후에만 실행
                node = msg.get("node", "")
                if node == "final_responder" or node == "Final_Responder":
                    logging.info("🎨 Final responder completed, triggering UI refresh for artifacts")
                    safe_rerun()
                    
        except Exception as e:
            logging.error(f"ArtifactStreamCallback error: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Return callback status information."""
        return {
            "known_artifacts": len(self._known_artifact_ids),
            "last_check": self._last_check,
            "check_interval": self._check_interval
        } 