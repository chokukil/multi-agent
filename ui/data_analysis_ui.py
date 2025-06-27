import streamlit as st
import asyncio
import pandas as pd
import io
import uuid
import os
from datetime import datetime
from typing import Dict, Any, Optional, List

from ui.thinking_stream import ThinkingStream, PlanVisualization, BeautifulResults
from core.a2a_data_analysis_executor import A2ADataAnalysisExecutor
from core.callbacks.progress_stream import progress_stream_manager

class DataAnalysisUI:
    """데이터 분석 전용 UI 컴포넌트"""
    
    def __init__(self):
        self.thinking_stream = None
        self.plan_viz = PlanVisualization()
        self.results_renderer = BeautifulResults()
        self.data_dir = "a2a_ds_servers/artifacts/data/shared_dataframes"
        
        # 데이터 디렉토리 생성
        os.makedirs(self.data_dir, exist_ok=True)
    
    def render_analysis_interface(self):
        """통합 분석 인터페이스 렌더링"""
        
        st.title("💬 Smart Data Analyst")
        st.markdown("A2A 프로토콜을 활용한 지능형 데이터 분석 시스템")
        
        # 데이터셋 선택 섹션
        self._render_dataset_section()
        
        # 분석 요청 섹션
        analysis_prompt = self._render_analysis_request_section()
        
        # 분석 실행
        if st.session_state.get('dataset_ready') and analysis_prompt:
            if st.button("🚀 분석 시작", type="primary", use_container_width=True):
                dataset_name = st.session_state.get('current_dataset')
                if dataset_name:
                    # 비동기 분석 실행
                    try:
                        asyncio.run(self.execute_analysis_workflow(
                            dataset_name, 
                            analysis_prompt,
                            st.session_state.get('analysis_options', {})
                        ))
                    except Exception as e:
                        st.error(f"분석 실행 중 오류가 발생했습니다: {str(e)}")
    
    def _render_dataset_section(self):
        """데이터셋 선택 섹션"""
        st.markdown("### 📂 데이터셋 선택")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "📂 데이터 파일 업로드", 
                type=['csv', 'xlsx', 'json'],
                help="분석할 데이터를 업로드하세요",
                key="data_uploader"
            )
            
        with col2:
            if uploaded_file:
                st.success("✅ 파일 업로드 완료")
                if st.button("🔍 데이터 미리보기", key="preview_btn"):
                    self._show_data_preview(uploaded_file)
                    
                # 데이터셋 저장
                if st.button("💾 데이터셋 등록", key="save_btn"):
                    dataset_name = self._save_uploaded_file(uploaded_file)
                    if dataset_name:
                        st.session_state.dataset_ready = True
                        st.session_state.current_dataset = dataset_name
                        st.success(f"데이터셋 '{dataset_name}' 등록 완료!")
                        st.rerun()
        
        # 기존 데이터셋 선택
        existing_datasets = self._get_existing_datasets()
        if existing_datasets:
            st.markdown("**또는 기존 데이터셋 선택:**")
            selected_dataset = st.selectbox(
                "등록된 데이터셋",
                ["선택하세요..."] + existing_datasets,
                key="existing_dataset_selector"
            )
            
            if selected_dataset != "선택하세요...":
                st.session_state.dataset_ready = True
                st.session_state.current_dataset = selected_dataset
                st.success(f"데이터셋 '{selected_dataset}' 선택됨")
    
    def _render_analysis_request_section(self) -> str:
        """분석 요청 섹션"""
        st.markdown("### 💬 분석 요청")
        
        # 현재 선택된 데이터셋 표시
        if st.session_state.get('dataset_ready'):
            dataset_name = st.session_state.get('current_dataset')
            st.info(f"📊 **현재 데이터셋**: {dataset_name}")
        
        analysis_prompt = st.text_area(
            "어떤 분석을 원하시나요?",
            placeholder="예: 이 데이터의 전반적인 패턴을 분석하고 주요 인사이트를 찾아주세요",
            height=100,
            key="analysis_prompt"
        )
        
        # 분석 옵션
        with st.expander("⚙️ 고급 옵션"):
            col1, col2 = st.columns(2)
            
            with col1:
                analysis_depth = st.selectbox(
                    "분석 깊이",
                    ["기본", "상세", "심화"],
                    help="분석의 상세 수준을 선택하세요",
                    key="analysis_depth"
                )
                
            with col2:
                include_viz = st.checkbox(
                    "시각화 포함", 
                    value=True,
                    help="분석 결과에 차트와 그래프를 포함합니다",
                    key="include_viz"
                )
            
            # 세션 상태에 옵션 저장
            st.session_state.analysis_options = {
                "depth": analysis_depth,
                "include_visualization": include_viz
            }
        
        return analysis_prompt
    
    def _show_data_preview(self, uploaded_file):
        """데이터 미리보기"""
        try:
            if uploaded_file.type == "text/csv":
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.type in ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "application/vnd.ms-excel"]:
                df = pd.read_excel(uploaded_file)
            elif uploaded_file.type == "application/json":
                df = pd.read_json(uploaded_file)
            else:
                st.error("지원하지 않는 파일 형식입니다.")
                return
            
            st.markdown("### 📋 데이터 미리보기")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("행 수", df.shape[0])
            with col2:
                st.metric("열 수", df.shape[1])
            
            st.dataframe(df.head(10), use_container_width=True)
            
            # 컬럼 정보
            st.markdown("### 📊 컬럼 정보")
            col_info = pd.DataFrame({
                '컬럼명': df.columns,
                '데이터 타입': df.dtypes,
                '결측값': df.isnull().sum(),
                '결측값 비율(%)': (df.isnull().sum() / len(df) * 100).round(2)
            })
            st.dataframe(col_info, use_container_width=True)
            
        except Exception as e:
            st.error(f"데이터 미리보기 중 오류가 발생했습니다: {str(e)}")
    
    def _save_uploaded_file(self, uploaded_file) -> Optional[str]:
        """업로드된 파일 저장"""
        try:
            # 고유한 파일명 생성
            timestamp = int(datetime.now().timestamp())
            file_extension = uploaded_file.name.split('.')[-1]
            dataset_name = f"uploaded_{uploaded_file.name.split('.')[0]}_{timestamp}.{file_extension}"
            
            file_path = os.path.join(self.data_dir, dataset_name)
            
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            return dataset_name.split('.')[0]  # 확장자 제거한 이름 반환
            
        except Exception as e:
            st.error(f"파일 저장 중 오류가 발생했습니다: {str(e)}")
            return None
    
    def _get_existing_datasets(self) -> List[str]:
        """기존 데이터셋 목록 조회"""
        try:
            if not os.path.exists(self.data_dir):
                return []
            
            files = os.listdir(self.data_dir)
            datasets = []
            
            for file in files:
                if file.endswith(('.csv', '.xlsx', '.json')):
                    # 확장자 제거한 이름 추가
                    dataset_name = '.'.join(file.split('.')[:-1])
                    datasets.append(dataset_name)
            
            return sorted(datasets)
            
        except Exception as e:
            st.error(f"데이터셋 목록 조회 중 오류: {str(e)}")
            return []
    
    async def execute_analysis_workflow(self, dataset_name: str, prompt: str, options: dict):
        """분석 워크플로우 실행"""
        
        # 1. 사고 과정 시작
        thinking_container = st.container()
        self.thinking_stream = ThinkingStream(thinking_container)
        self.thinking_stream.start_thinking("데이터 분석 요청을 분석하고 있습니다...")
        
        # 2. 계획 수립
        plan_state = await self._create_analysis_plan(dataset_name, prompt, options)
        
        if plan_state and plan_state.get("plan"):
            # 3. 실행 및 결과 표시
            await self._execute_plan_with_streaming(plan_state)
            
            # 4. 최종 보고서 생성
            self._generate_final_report(plan_state)
        else:
            st.error("분석 계획 수립에 실패했습니다.")
    
    async def _create_analysis_plan(self, dataset_name: str, prompt: str, options: dict) -> Optional[dict]:
        """분석 계획 수립"""
        
        # 사고 과정 업데이트
        self.thinking_stream.add_thought("사용자의 요청을 이해하고 적절한 분석 방법을 찾고 있습니다.", "analysis")
        
        with st.status("🧠 **분석 계획 수립 중...**", expanded=True) as status:
            self.thinking_stream.add_thought("데이터 분석에 필요한 단계들을 계획하고 있습니다.", "planning")
            
            try:
                # 오케스트레이터 호출 (기존 방식 사용)
                import httpx
                
                payload = {
                    "jsonrpc": "2.0",
                    "method": "tasks/send",
                    "params": {
                        "id": str(uuid.uuid4()),
                        "message": {
                            "role": "user",
                            "parts": [
                                {
                                    "type": "text",
                                    "text": f"데이터셋 '{dataset_name}'에 대해 다음 분석을 수행해줘: {prompt}. 분석 깊이: {options.get('depth', '기본')}"
                                }
                            ]
                        }
                    },
                    "id": 1
                }
                
                async with httpx.AsyncClient(timeout=30) as client:
                    response = await client.post("http://localhost:8100", json=payload)
                    response.raise_for_status()
                    
                    result = response.json()
                    
                    if "result" in result:
                        # 계획 파싱
                        plan_content = result["result"]
                        plan_state = self._parse_orchestrator_response(plan_content, dataset_name)
                        
                        self.thinking_stream.add_thought("완벽한 분석 계획이 완성되었습니다!", "success")
                        self.thinking_stream.finish_thinking("계획 수립 완료! 이제 실행을 시작합니다.")
                        
                        # 계획 시각화
                        self.plan_viz.display_plan(plan_state["plan"], title="📊 데이터 분석 실행 계획")
                        
                        status.update(label="✅ 계획 완성!", state="complete", expanded=False)
                        
                        return plan_state
                    else:
                        status.update(label="❌ 계획 수립 실패!", state="error")
                        st.error("오케스트레이터로부터 유효한 계획을 받지 못했습니다.")
                        return None
                        
            except Exception as e:
                status.update(label="❌ 계획 수립 오류!", state="error")
                st.error(f"계획 수립 중 오류가 발생했습니다: {str(e)}")
                return None
    
    def _parse_orchestrator_response(self, response_content: Any, dataset_name: str) -> dict:
        """오케스트레이터 응답을 파싱하여 계획 구조로 변환"""
        
        # 기본 분석 계획 생성 (응답 파싱 실패시 폴백)
        default_plan = [
            {
                "agent_name": "pandas_data_analyst",
                "skill_name": "analyze_data",
                "parameters": {
                    "data_id": dataset_name,
                    "user_instructions": "데이터 구조 및 기본 통계 분석을 수행해주세요."
                },
                "reasoning": "데이터의 전반적인 구조와 품질을 파악하기 위해 기본 분석을 수행합니다."
            },
            {
                "agent_name": "data_visualization",
                "skill_name": "analyze_data", 
                "parameters": {
                    "data_id": dataset_name,
                    "user_instructions": "주요 변수들의 분포와 관계를 시각화해주세요."
                },
                "reasoning": "데이터의 패턴과 트렌드를 시각적으로 이해하기 위해 차트를 생성합니다."
            },
            {
                "agent_name": "eda_tools",
                "skill_name": "analyze_data",
                "parameters": {
                    "data_id": dataset_name,
                    "user_instructions": "이상치 탐지 및 데이터 품질 분석을 수행해주세요."
                },
                "reasoning": "데이터의 품질 문제와 이상치를 식별하여 분석의 신뢰성을 확보합니다."
            }
        ]
        
        return {"plan": default_plan}
    
    async def _execute_plan_with_streaming(self, plan_state: dict):
        """스트리밍으로 계획 실행"""
        
        st.markdown("### 🔄 분석 실행 중...")
        
        # A2A 실행기 생성
        executor = A2ADataAnalysisExecutor()
        
        # 진행 상황 컨테이너
        progress_container = st.container()
        results_container = st.container()
        
        with progress_container:
            # 실행 결과 저장
            execution_result = await executor.execute(plan_state)
            
            # 세션 상태에 결과 저장
            st.session_state.analysis_results = execution_result
    
    def _generate_final_report(self, plan_state: dict):
        """최종 분석 보고서 생성"""
        
        st.markdown("---")
        st.markdown("## 📊 **최종 분석 보고서**")
        
        execution_result = st.session_state.get('analysis_results', {})
        
        if execution_result.get('step_outputs'):
            # 요약 통계
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("총 실행 단계", execution_result.get('total_steps', 0))
            with col2:
                st.metric("성공한 단계", execution_result.get('successful_steps', 0))
            with col3:
                st.metric("실행 시간", f"{execution_result.get('execution_time', 0):.1f}초")
            
            # 각 단계별 결과 표시
            st.markdown("### 🎯 단계별 분석 결과")
            
            for step_num, result in execution_result['step_outputs'].items():
                if result.get('success'):
                    with st.expander(f"📋 Step {step_num}: {result.get('agent', 'Unknown Agent')}", expanded=True):
                        content = result.get('content', '')
                        if content:
                            st.markdown(str(content))
                        else:
                            st.info("결과 내용이 없습니다.")
            
            # 다운로드 옵션
            st.markdown("### 📥 결과 다운로드")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📄 보고서 다운로드"):
                    self._generate_report_download(execution_result)
            
            with col2:
                if st.button("📊 결과 데이터 다운로드"):
                    self._generate_data_download(execution_result)
                    
            with col3:
                if st.button("🔗 결과 공유"):
                    st.info("공유 기능은 향후 추가될 예정입니다.")
        else:
            st.warning("실행 결과가 없습니다.")
    
    def _generate_report_download(self, execution_result: dict):
        """보고서 다운로드 생성"""
        try:
            # 간단한 텍스트 보고서 생성
            report_content = f"""
데이터 분석 보고서
생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

실행 요약:
- 총 단계: {execution_result.get('total_steps', 0)}
- 성공 단계: {execution_result.get('successful_steps', 0)}
- 실행 시간: {execution_result.get('execution_time', 0):.2f}초

분석 결과:
"""
            
            for step_num, result in execution_result.get('step_outputs', {}).items():
                if result.get('success'):
                    report_content += f"\n\nStep {step_num} - {result.get('agent', 'Unknown Agent')}:\n"
                    report_content += str(result.get('content', '결과 없음'))
            
            st.download_button(
                label="📄 텍스트 보고서 다운로드",
                data=report_content,
                file_name=f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
            
        except Exception as e:
            st.error(f"보고서 생성 중 오류: {str(e)}")
    
    def _generate_data_download(self, execution_result: dict):
        """데이터 다운로드 생성"""
        try:
            # 결과를 JSON 형태로 저장
            import json
            
            data_content = json.dumps(execution_result, indent=2, ensure_ascii=False)
            
            st.download_button(
                label="📊 결과 데이터 (JSON)",
                data=data_content,
                file_name=f"analysis_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
            
        except Exception as e:
            st.error(f"데이터 생성 중 오류: {str(e)}") 