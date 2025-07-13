"""
통합 테스트 - A2A 에이전트 간 협업 및 전체 워크플로우 테스트
"""
import pytest
import asyncio
import streamlit as st
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import tempfile
import os
import time
from datetime import datetime

# 테스트 대상 모듈 임포트
try:
    from ui.layout.split_layout import create_split_layout
    from ui.components.file_upload import create_file_upload_manager
    from ui.components.chat_interface import create_chat_interface
    from ui.components.question_input import create_question_input
    from core.shared_knowledge_bank import create_shared_knowledge_bank
    COMPONENTS_AVAILABLE = True
except ImportError:
    COMPONENTS_AVAILABLE = False

class TestWorkflowIntegration:
    """워크플로우 통합 테스트"""
    
    def setup_method(self):
        """테스트 설정"""
        if not COMPONENTS_AVAILABLE:
            pytest.skip("Components not available")
        
        # 세션 상태 초기화
        st.session_state = {}
        
        # 컴포넌트 초기화
        self.split_layout = create_split_layout()
        self.file_manager = create_file_upload_manager()
        self.chat_interface = create_chat_interface()
        self.question_input = create_question_input()
    
    def test_complete_data_analysis_workflow(self):
        """완전한 데이터 분석 워크플로우 테스트"""
        # 1. 파일 업로드 시뮬레이션
        mock_file = Mock()
        mock_file.name = "test_data.csv"
        mock_file.size = 1024
        mock_file.type = "text/csv"
        
        file_info = self.file_manager.get_file_info(mock_file)
        assert file_info['is_data'] == True
        
        # 2. 질문 입력
        question = "이 데이터를 분석해주세요"
        assert self.question_input._validate_input(question) == True
        
        # 3. 채팅 메시지 추가
        self.chat_interface.add_user_message(question)
        
        # 4. 오케스트레이터 계획 시뮬레이션
        plan_metadata = {
            'type': 'orchestrator_plan',
            'plan': {
                'tasks': [
                    {'description': '데이터 로드', 'status': 'completed'},
                    {'description': '분석 수행', 'status': 'in_progress'}
                ]
            }
        }
        self.chat_interface.add_assistant_message("계획을 수립했습니다", plan_metadata)
        
        # 5. 에이전트 상태 업데이트
        agent_metadata = {
            'type': 'agent_status',
            'agent_name': 'Pandas Agent',
            'status': '데이터 처리 중',
            'details': '1024 rows loaded'
        }
        self.chat_interface.add_assistant_message("", agent_metadata)
        
        # 6. 아티팩트 생성
        artifacts_metadata = {
            'type': 'artifacts',
            'artifacts': [
                {'id': 'summary', 'name': '데이터 요약', 'icon': '📊'},
                {'id': 'chart', 'name': '분석 차트', 'icon': '📈'}
            ]
        }
        self.chat_interface.add_assistant_message("아티팩트가 생성되었습니다", artifacts_metadata)
        
        # 7. 최종 답변
        final_answer = "## 분석 결과\n데이터 분석이 완료되었습니다."
        self.chat_interface.add_assistant_message(final_answer)
        
        # 8. 워크플로우 검증
        messages = self.chat_interface.get_messages()
        assert len(messages) == 5  # 사용자 질문 + 4개 응답
        
        # 메시지 타입 검증
        assert messages[0].role == "🧑🏻"
        assert messages[1].metadata.get('type') == 'orchestrator_plan'
        assert messages[2].metadata.get('type') == 'agent_status'
        assert messages[3].metadata.get('type') == 'artifacts'
        assert "분석 결과" in messages[4].content
    
    def test_multi_agent_collaboration(self):
        """멀티 에이전트 협업 테스트"""
        # 에이전트 시뮬레이션 데이터
        agents = [
            {'name': 'Orchestrator', 'role': '🍒', 'task': '계획 수립'},
            {'name': 'Pandas Agent', 'role': '🍒', 'task': '데이터 로드'},
            {'name': 'Analysis Agent', 'role': '🍒', 'task': '통계 분석'},
            {'name': 'Viz Agent', 'role': '🍒', 'task': '시각화 생성'}
        ]
        
        # 협업 시뮬레이션
        for i, agent in enumerate(agents):
            status_metadata = {
                'type': 'agent_status',
                'agent_name': agent['name'],
                'status': f"{agent['task']} 완료",
                'details': f"Step {i+1} completed"
            }
            
            self.chat_interface.add_assistant_message(
                f"{agent['name']}: {agent['task']} 완료",
                status_metadata
            )
        
        # 협업 결과 검증
        messages = self.chat_interface.get_messages()
        assert len(messages) == len(agents)
        
        # 각 에이전트 메시지 검증
        for i, message in enumerate(messages):
            assert message.role == agents[i]['role']
            assert message.metadata.get('type') == 'agent_status'
            assert agents[i]['name'] in message.metadata.get('agent_name', '')
    
    def test_split_layout_integration(self):
        """Split Layout 통합 테스트"""
        # 좌측 패널 컨텐츠 함수
        def left_content():
            self.chat_interface.add_user_message("좌측 패널 테스트")
            return "Left Panel Content"
        
        # 우측 패널 컨텐츠 함수  
        def right_content():
            return "Right Panel Content"
        
        # CSS 생성 테스트
        css = self.split_layout._get_split_css(0.3)
        assert ".split-container" in css
        assert "30%" in css
        assert "70%" in css
        
        # JavaScript 생성 테스트
        js = self.split_layout._get_split_js()
        assert "initializeSplitLayout" in js
        assert "handleMouseDown" in js
        assert "updateLayout" in js
    
    def test_error_handling_integration(self):
        """에러 처리 통합 테스트"""
        # 잘못된 파일 형식 처리
        mock_file = Mock()
        mock_file.name = "test.xyz"  # 지원되지 않는 확장자
        mock_file.size = 1024
        mock_file.type = "application/unknown"
        
        file_info = self.file_manager.get_file_info(mock_file)
        assert file_info['is_data'] == False
        assert file_info['is_image'] == False
        
        # 잘못된 질문 입력 처리
        invalid_inputs = ["", "  ", "a", "spam test"]
        
        for invalid_input in invalid_inputs:
            assert self.question_input._validate_input(invalid_input) == False
        
        # 빈 메시지 처리
        initial_count = len(self.chat_interface.get_messages())
        self.chat_interface.add_assistant_message("")  # 빈 메시지
        
        messages = self.chat_interface.get_messages()
        assert len(messages) == initial_count + 1
        assert messages[-1].content == ""

class TestKnowledgeBankIntegration:
    """지식 은행 통합 테스트"""
    
    def setup_method(self):
        """테스트 설정"""
        if not COMPONENTS_AVAILABLE:
            pytest.skip("Components not available")
        
        # 임시 데이터베이스 사용
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        
        # 지식 은행 초기화 (실제 데이터베이스 사용 안함)
        self.knowledge_bank = None
        try:
            from core.shared_knowledge_bank import SharedKnowledgeBank
            self.knowledge_bank = SharedKnowledgeBank(self.temp_db.name)
        except Exception:
            # 의존성이 없으면 Mock 사용
            self.knowledge_bank = Mock()
    
    def teardown_method(self):
        """테스트 정리"""
        if os.path.exists(self.temp_db.name):
            os.unlink(self.temp_db.name)
    
    def test_knowledge_accumulation_workflow(self):
        """지식 축적 워크플로우 테스트"""
        if not hasattr(self.knowledge_bank, 'add_knowledge'):
            pytest.skip("Knowledge bank not available")
        
        # 1. 질문-답변 세션 시뮬레이션
        qa_pairs = [
            ("데이터 분석 방법은?", "EDA, 시각화, 모델링 순서로 진행"),
            ("결측값 처리는?", "삭제, 대치, 예측 방법 중 선택"),
            ("모델 평가는?", "정확도, 정밀도, 재현율 등 지표 사용")
        ]
        
        # 2. 지식 추가
        for question, answer in qa_pairs:
            knowledge_content = f"Q: {question}\nA: {answer}"
            node_id = self.knowledge_bank.add_knowledge(
                knowledge_content, 
                "insight",
                {"source": "qa_session"}
            )
            assert node_id is not None
        
        # 3. 지식 검색 테스트
        search_results = self.knowledge_bank.search_knowledge("데이터 분석", "embedding")
        assert len(search_results) > 0
        
        # 4. 통계 확인
        stats = self.knowledge_bank.get_knowledge_stats()
        assert stats['total_nodes'] >= 3
        assert 'insight' in stats['node_types']
    
    def test_knowledge_search_integration(self):
        """지식 검색 통합 테스트"""
        if not hasattr(self.knowledge_bank, 'search_knowledge'):
            pytest.skip("Knowledge bank not available")
        
        # 테스트 지식 추가
        test_knowledge = [
            ("머신러닝 기초", "supervised learning includes classification and regression"),
            ("데이터 전처리", "clean data, handle missing values, normalize features"),
            ("모델 평가", "use cross-validation and multiple metrics")
        ]
        
        for title, content in test_knowledge:
            self.knowledge_bank.add_knowledge(content, "rule", {"title": title})
        
        # 임베딩 검색 테스트
        results = self.knowledge_bank.search_knowledge("machine learning", "embedding")
        assert isinstance(results, list)
        
        # 그래프 검색 테스트  
        results = self.knowledge_bank.search_knowledge("evaluation", "graph")
        assert isinstance(results, list)
    
    @patch('sqlite3.connect')
    def test_knowledge_persistence(self, mock_connect):
        """지식 영속성 테스트"""
        if not hasattr(self.knowledge_bank, 'save_node_to_db'):
            pytest.skip("Knowledge bank not available")
        
        # Mock 데이터베이스 연결
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_connect.return_value.__enter__.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        
        # 지식 추가 및 저장 테스트
        node_id = self.knowledge_bank.add_knowledge(
            "Test knowledge for persistence",
            "fact",
            {"test": True}
        )
        
        # 데이터베이스 호출 확인
        assert mock_connect.called
        assert mock_cursor.execute.called

class TestPerformanceIntegration:
    """성능 통합 테스트"""
    
    def setup_method(self):
        """테스트 설정"""
        if not COMPONENTS_AVAILABLE:
            pytest.skip("Components not available")
        
        st.session_state = {}
        self.chat_interface = create_chat_interface()
    
    def test_large_message_handling(self):
        """대용량 메시지 처리 테스트"""
        # 대용량 메시지 생성
        large_content = "A" * 10000  # 10KB 텍스트
        
        start_time = time.time()
        self.chat_interface.add_user_message(large_content)
        end_time = time.time()
        
        # 처리 시간 확인 (1초 이내)
        assert (end_time - start_time) < 1.0
        
        # 메시지 저장 확인
        messages = self.chat_interface.get_messages()
        assert len(messages) == 1
        assert len(messages[0].content) == 10000
    
    def test_multiple_messages_performance(self):
        """다중 메시지 성능 테스트"""
        message_count = 100
        
        start_time = time.time()
        for i in range(message_count):
            self.chat_interface.add_assistant_message(f"메시지 {i}")
        end_time = time.time()
        
        # 처리 시간 확인 (2초 이내)
        assert (end_time - start_time) < 2.0
        
        # 메시지 개수 확인
        messages = self.chat_interface.get_messages()
        assert len(messages) == message_count
    
    def test_memory_usage_stability(self):
        """메모리 사용량 안정성 테스트"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # 대량 작업 수행
        for i in range(1000):
            self.chat_interface.add_assistant_message(f"Memory test {i}")
            
            if i % 100 == 0:
                # 메시지 일부 정리
                messages = self.chat_interface.get_messages()
                if len(messages) > 50:
                    # 세션 상태에서 일부 메시지 제거 (실제 구현에서는 다른 방식 사용)
                    st.session_state[self.chat_interface.messages_key] = messages[-50:]
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # 메모리 증가량이 100MB 이내인지 확인
        assert memory_increase < 100 * 1024 * 1024

class TestEndToEndIntegration:
    """End-to-End 통합 테스트"""
    
    def setup_method(self):
        """테스트 설정"""
        if not COMPONENTS_AVAILABLE:
            pytest.skip("Components not available")
        
        st.session_state = {}
        
        # 전체 시스템 컴포넌트 초기화
        self.components = {
            'split_layout': create_split_layout(),
            'file_manager': create_file_upload_manager(),
            'chat_interface': create_chat_interface(),
            'question_input': create_question_input()
        }
    
    def test_complete_user_journey(self):
        """완전한 사용자 여정 테스트"""
        # 1. 사용자가 파일 업로드
        mock_file = Mock()
        mock_file.name = "sales_data.csv"
        mock_file.size = 2048
        mock_file.type = "text/csv"
        
        file_info = self.components['file_manager'].get_file_info(mock_file)
        assert file_info['is_data'] == True
        
        # 2. 사용자가 질문 입력
        question = "매출 데이터의 트렌드를 분석해주세요"
        assert self.components['question_input']._validate_input(question) == True
        
        # 3. 질문 히스토리에 추가
        self.components['question_input'].add_to_history(question)
        history = self.components['question_input'].get_input_history()
        assert question in history
        
        # 4. 채팅에 사용자 메시지 추가
        self.components['chat_interface'].add_user_message(question)
        
        # 5. 시스템 응답 시뮬레이션
        responses = [
            ("계획 수립 중...", {'type': 'orchestrator_plan', 'plan': {'tasks': []}}),
            ("데이터 로드 중...", {'type': 'agent_status', 'agent_name': 'Pandas Agent', 'status': 'loading'}),
            ("분석 수행 중...", {'type': 'agent_status', 'agent_name': 'Analysis Agent', 'status': 'analyzing'}),
            ("결과 생성됨", {'type': 'artifacts', 'artifacts': [{'id': 'trend', 'name': '트렌드 차트'}]}),
            ("## 매출 트렌드 분석 결과\n상승 추세입니다.", {})
        ]
        
        for content, metadata in responses:
            self.components['chat_interface'].add_assistant_message(content, metadata)
        
        # 6. 전체 대화 검증
        messages = self.components['chat_interface'].get_messages()
        assert len(messages) == 6  # 사용자 질문 + 5개 응답
        
        # 7. Split Layout 기능 테스트
        css = self.components['split_layout']._get_split_css(0.3)
        assert "split-container" in css
        
        # 8. 마지막 메시지가 최종 답변인지 확인
        last_message = self.components['chat_interface'].get_last_message()
        assert "매출 트렌드" in last_message.content
        assert "상승 추세" in last_message.content
    
    def test_error_recovery_workflow(self):
        """에러 복구 워크플로우 테스트"""
        # 1. 잘못된 파일 업로드 시도
        invalid_file = Mock()
        invalid_file.name = "document.pdf"
        invalid_file.size = 1024
        invalid_file.type = "application/pdf"
        
        file_info = self.components['file_manager'].get_file_info(invalid_file)
        assert file_info['is_data'] == False
        assert file_info['is_image'] == False
        
        # 2. 잘못된 질문 입력
        invalid_question = "a"
        assert self.components['question_input']._validate_input(invalid_question) == False
        
        # 3. 올바른 질문 재입력
        valid_question = "데이터를 분석해주세요"
        assert self.components['question_input']._validate_input(valid_question) == True
        
        # 4. 시스템 에러 시뮬레이션 및 복구
        self.components['chat_interface'].add_user_message(valid_question)
        self.components['chat_interface'].add_assistant_message(
            "처리 중 오류가 발생했습니다. 다시 시도합니다.",
            {'type': 'error', 'error_code': 'processing_error'}
        )
        self.components['chat_interface'].add_assistant_message(
            "분석이 완료되었습니다.",
            {'type': 'success'}
        )
        
        # 5. 복구 확인
        messages = self.components['chat_interface'].get_messages()
        assert len(messages) == 3
        assert messages[-1].metadata.get('type') == 'success'
    
    def test_concurrent_operations(self):
        """동시 작업 테스트"""
        # 여러 질문을 빠르게 연속 입력
        questions = [
            "첫 번째 질문입니다",
            "두 번째 질문입니다", 
            "세 번째 질문입니다"
        ]
        
        for question in questions:
            self.components['chat_interface'].add_user_message(question)
            self.components['question_input'].add_to_history(question)
        
        # 모든 질문이 올바르게 저장되었는지 확인
        messages = self.components['chat_interface'].get_messages()
        history = self.components['question_input'].get_input_history()
        
        assert len(messages) == 3
        assert len(history) == 3
        
        # 순서 확인 (최신 순)
        assert history[0] == questions[-1]
        assert history[1] == questions[-2]
        assert history[2] == questions[-3]

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 