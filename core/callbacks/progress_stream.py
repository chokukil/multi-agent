# core/callbacks/progress_stream.py
import streamlit as st
from typing import Dict, Any, List
import logging

class ProgressStreamCallback:
    """Handles streaming of plan and progress to the UI expander."""

    def __init__(self, expander_placeholder):
        self.placeholder = expander_placeholder
        self.plan: List[Dict] = []
        self.current_step = 0
        self.total_steps = 0
        self.plan_displayed = False
        self.completed_steps = set()

    def __call__(self, msg: Dict[str, Any]):
        node = msg.get("node", "")
        content = msg.get("content", "")

        try:
            if node == "planner" and isinstance(content, dict) and "plan" in content:
                if not self.plan_displayed:
                    self.plan = content["plan"]
                    self.total_steps = len(self.plan)
                    self.plan_displayed = True
                    self.current_step = 0
                    logging.info(f"📋 Plan loaded: {self.total_steps} steps")
                    self.render_progress()
            
            elif node == "router":
                # 라우터는 다음 단계를 결정
                self.current_step = st.session_state.get("current_step", 0)
                logging.info(f"🔀 Router: Current step {self.current_step + 1}")
                self.render_progress()

            elif "executor" in node.lower() or node in ["Data_Validator", "Preprocessing_Expert", "EDA_Analyst", "Visualization_Expert", "ML_Specialist", "Statistical_Analyst", "Report_Generator"]:
                # Executor 완료 확인
                if isinstance(content, str) and "TASK COMPLETED:" in content:
                    completed_step = st.session_state.get("current_step", 0)
                    self.completed_steps.add(completed_step)
                    logging.info(f"✅ Step {completed_step + 1} completed by {node}")
                    self.render_progress()

            elif node == "final_responder" or node == "Final_Responder":
                self.current_step = self.total_steps # Mark as fully complete
                # 모든 단계를 완료로 표시
                for i in range(self.total_steps):
                    self.completed_steps.add(i)
                logging.info("🎯 Final responder: All steps completed")
                self.render_progress(completed=True)
            
            elif node == "direct_response":
                # 직접 응답의 경우 즉시 완료로 표시
                logging.info("⚡ Direct response: Task completed immediately")
                self.placeholder.success("✅ 질문 처리 완료 (직접 응답)")

        except Exception as e:
            logging.error(f"Error in ProgressStreamCallback: {e}")

    def render_progress(self, completed: bool = False):
        if not self.plan:
            return

        try:
            title = f"📋 진행 현황 ({len(self.completed_steps)}/{self.total_steps})"
            if completed:
                title = f"✅ 분석 완료 ({self.total_steps}/{self.total_steps})"

            with self.placeholder:
                st.empty() # Clear previous content
                with st.expander(title, expanded=not completed):
                    for i, step in enumerate(self.plan):
                        step_num = i + 1
                        task = step.get('task', 'Unknown Task')
                        
                        # 완료된 단계 체크
                        if i in self.completed_steps:
                            st.markdown(f"✅ ~~{step_num}. {task}~~")
                        elif step_num == self.current_step + 1:
                            st.markdown(f"▶️ **{step_num}. {task}**")
                        else:
                            st.markdown(f"⚪️ {step_num}. {task}")
            
            logging.info(f"🔄 ProgressStream: Updated to step {self.current_step + 1}/{self.total_steps}, completed: {len(self.completed_steps)}")
            
        except Exception as e:
            logging.error(f"Error rendering progress: {e}") 