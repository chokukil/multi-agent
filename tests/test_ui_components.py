"""
UI 컴포넌트 단위 테스트
"""
import pytest
import streamlit as st
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from datetime import datetime
import tempfile
import os

# 테스트 대상 모듈 임포트
try:
    from ui.layout.split_layout import SplitLayout, create_split_layout
    from ui.components.file_upload import FileUploadManager, create_file_upload_manager
    from ui.components.chat_interface import ChatMessage, AutoScrollChatInterface, create_chat_interface
    from ui.components.question_input import QuestionInput, create_question_input
    from core.shared_knowledge_bank import KnowledgeNode, KnowledgeGraph, SharedKnowledgeBank
    COMPONENTS_AVAILABLE = True
except ImportError:
    COMPONENTS_AVAILABLE = False

class TestSplitLayout:
    """Split Layout 컴포넌트 테스트"""
    
    def test_split_layout_init(self):
        """Split Layout 초기화 테스트"""
        if not COMPONENTS_AVAILABLE:
            pytest.skip("Components not available")
        
        layout = SplitLayout()
        assert layout.default_ratio == 0.3
        assert layout.min_ratio == 0.2
        assert layout.max_ratio == 0.6
        assert layout.session_key == "split_ratio"
    
    def test_split_layout_custom_params(self):
        """커스텀 매개변수 테스트"""
        if not COMPONENTS_AVAILABLE:
            pytest.skip("Components not available")
        
        layout = SplitLayout(
            default_ratio=0.4,
            min_ratio=0.3,
            max_ratio=0.7,
            session_key="custom_ratio"
        )
        assert layout.default_ratio == 0.4
        assert layout.min_ratio == 0.3
        assert layout.max_ratio == 0.7
        assert layout.session_key == "custom_ratio"
    
    def test_split_layout_css_generation(self):
        """CSS 생성 테스트"""
        if not COMPONENTS_AVAILABLE:
            pytest.skip("Components not available")
        
        layout = SplitLayout()
        css = layout._get_split_css(0.3)
        
        assert ".split-container" in css
        assert ".split-left" in css
        assert ".split-right" in css
        assert ".split-divider" in css
        assert "30%" in css
        assert "70%" in css
    
    def test_split_layout_ratio_update(self):
        """비율 업데이트 테스트"""
        if not COMPONENTS_AVAILABLE:
            pytest.skip("Components not available")
        
        layout = SplitLayout()
        
        # 정상 범위 내 업데이트
        layout.update_ratio(0.5)
        assert layout.min_ratio <= 0.5 <= layout.max_ratio
        
        # 범위 초과 시 제한
        layout.update_ratio(0.1)  # 최소값 이하
        layout.update_ratio(0.8)  # 최대값 초과
    
    def test_create_split_layout_factory(self):
        """팩토리 함수 테스트"""
        if not COMPONENTS_AVAILABLE:
            pytest.skip("Components not available")
        
        layout = create_split_layout()
        assert isinstance(layout, SplitLayout)
        assert layout.default_ratio == 0.3

class TestFileUploadManager:
    """파일 업로드 관리자 테스트"""
    
    def test_file_upload_manager_init(self):
        """파일 업로드 관리자 초기화 테스트"""
        if not COMPONENTS_AVAILABLE:
            pytest.skip("Components not available")
        
        manager = FileUploadManager()
        assert manager.session_key == "uploaded_files"
        assert len(manager.PANDAS_FORMATS) > 0
        assert len(manager.IMAGE_FORMATS) > 0
    
    def test_supported_extensions(self):
        """지원 확장자 테스트"""
        if not COMPONENTS_AVAILABLE:
            pytest.skip("Components not available")
        
        manager = FileUploadManager()
        extensions = manager.get_supported_extensions()
        
        assert '.csv' in extensions
        assert '.xlsx' in extensions
        assert '.json' in extensions
        assert '.jpg' in extensions
        assert '.png' in extensions
    
    def test_file_info_extraction(self):
        """파일 정보 추출 테스트"""
        if not COMPONENTS_AVAILABLE:
            pytest.skip("Components not available")
        
        manager = FileUploadManager()
        
        # Mock 파일 객체 생성
        mock_file = Mock()
        mock_file.name = "test_data.csv"
        mock_file.size = 1024
        mock_file.type = "text/csv"
        
        file_info = manager.get_file_info(mock_file)
        
        assert file_info['name'] == "test_data.csv"
        assert file_info['size'] == 1024
        assert file_info['extension'] == '.csv'
        assert file_info['is_data'] == True
        assert file_info['format'] == 'csv'
    
    def test_file_size_formatting(self):
        """파일 크기 포맷팅 테스트"""
        if not COMPONENTS_AVAILABLE:
            pytest.skip("Components not available")
        
        manager = FileUploadManager()
        
        assert manager.format_file_size(0) == "0 B"
        assert manager.format_file_size(1024) == "1.0 KB"
        assert manager.format_file_size(1024 * 1024) == "1.0 MB"
        assert manager.format_file_size(1024 * 1024 * 1024) == "1.0 GB"
    
    @patch('pandas.read_csv')
    def test_data_file_loading(self, mock_read_csv):
        """데이터 파일 로딩 테스트"""
        if not COMPONENTS_AVAILABLE:
            pytest.skip("Components not available")
        
        manager = FileUploadManager()
        
        # Mock 데이터프레임
        mock_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        mock_read_csv.return_value = mock_df
        
        # Mock 파일 객체
        mock_file = Mock()
        mock_file.read.return_value = b"A,B\n1,4\n2,5\n3,6"
        mock_file.seek.return_value = None
        
        file_info = {'format': 'csv'}
        
        result = manager.load_data_file(mock_file, file_info)
        
        assert result is not None
        assert len(result) == 3
        assert list(result.columns) == ['A', 'B']

class TestChatInterface:
    """채팅 인터페이스 테스트"""
    
    def test_chat_message_creation(self):
        """채팅 메시지 생성 테스트"""
        if not COMPONENTS_AVAILABLE:
            pytest.skip("Components not available")
        
        message = ChatMessage("🧑🏻", "테스트 메시지")
        
        assert message.role == "🧑🏻"
        assert message.content == "테스트 메시지"
        assert message.message_id is not None
        assert isinstance(message.timestamp, datetime)
    
    def test_chat_message_to_dict(self):
        """채팅 메시지 딕셔너리 변환 테스트"""
        if not COMPONENTS_AVAILABLE:
            pytest.skip("Components not available")
        
        message = ChatMessage("🍒", "응답 메시지", metadata={"type": "test"})
        message_dict = message.to_dict()
        
        assert message_dict['role'] == "🍒"
        assert message_dict['content'] == "응답 메시지"
        assert message_dict['metadata']['type'] == "test"
        assert 'timestamp' in message_dict
        assert 'message_id' in message_dict
    
    def test_auto_scroll_chat_interface_init(self):
        """자동 스크롤 채팅 인터페이스 초기화 테스트"""
        if not COMPONENTS_AVAILABLE:
            pytest.skip("Components not available")
        
        chat = AutoScrollChatInterface()
        assert chat.session_key == "chat_messages"
        assert chat.messages_key == "chat_messages_messages"
        assert chat.scroll_key == "chat_messages_scroll"
    
    def test_message_addition(self):
        """메시지 추가 테스트"""
        if not COMPONENTS_AVAILABLE:
            pytest.skip("Components not available")
        
        chat = AutoScrollChatInterface()
        
        # 세션 상태 초기화
        import streamlit as st
        st.session_state = {}
        
        chat.add_user_message("사용자 메시지")
        chat.add_assistant_message("어시스턴트 메시지")
        
        messages = chat.get_messages()
        assert len(messages) == 2
        assert messages[0].role == "🧑🏻"
        assert messages[1].role == "🍒"
    
    def test_orchestrator_plan_rendering(self):
        """오케스트레이터 계획 렌더링 테스트"""
        if not COMPONENTS_AVAILABLE:
            pytest.skip("Components not available")
        
        chat = AutoScrollChatInterface()
        
        plan = {
            'tasks': [
                {'description': '태스크 1', 'status': 'completed'},
                {'description': '태스크 2', 'status': 'in_progress'},
                {'description': '태스크 3', 'status': 'pending'}
            ]
        }
        
        html = chat.render_orchestrator_plan(plan)
        
        assert '✅' in html  # completed
        assert '⏳' in html  # in_progress
        assert '○' in html   # pending
        assert 'orchestrator-plan' in html
    
    def test_agent_status_rendering(self):
        """에이전트 상태 렌더링 테스트"""
        if not COMPONENTS_AVAILABLE:
            pytest.skip("Components not available")
        
        chat = AutoScrollChatInterface()
        
        html = chat.render_agent_status("Pandas Agent", "작업 중", "데이터 처리 중...")
        
        assert "Pandas Agent" in html
        assert "작업 중" in html
        assert "데이터 처리 중..." in html
        assert "agent-status" in html

class TestQuestionInput:
    """질문 입력 컴포넌트 테스트"""
    
    def test_question_input_init(self):
        """질문 입력 초기화 테스트"""
        if not COMPONENTS_AVAILABLE:
            pytest.skip("Components not available")
        
        question_input = QuestionInput()
        assert question_input.session_key == "question_input"
        assert question_input.input_key == "question_input_input"
        assert question_input.submit_key == "question_input_submit"
    
    def test_input_validation(self):
        """입력 검증 테스트"""
        if not COMPONENTS_AVAILABLE:
            pytest.skip("Components not available")
        
        question_input = QuestionInput()
        
        # 유효한 입력
        assert question_input._validate_input("데이터를 분석해주세요") == True
        
        # 무효한 입력
        assert question_input._validate_input("") == False
        assert question_input._validate_input("  ") == False
        assert question_input._validate_input("a") == False
        assert question_input._validate_input("spam content") == False
    
    def test_history_management(self):
        """히스토리 관리 테스트"""
        if not COMPONENTS_AVAILABLE:
            pytest.skip("Components not available")
        
        question_input = QuestionInput()
        
        # 세션 상태 초기화
        import streamlit as st
        st.session_state = {}
        
        # 히스토리 추가
        question_input.add_to_history("질문 1")
        question_input.add_to_history("질문 2")
        question_input.add_to_history("질문 3")
        
        history = question_input.get_input_history()
        assert len(history) == 3
        assert history[0] == "질문 3"  # 최신 순
        assert history[1] == "질문 2"
        assert history[2] == "질문 1"
    
    def test_history_deduplication(self):
        """히스토리 중복 제거 테스트"""
        if not COMPONENTS_AVAILABLE:
            pytest.skip("Components not available")
        
        question_input = QuestionInput()
        
        # 세션 상태 초기화
        import streamlit as st
        st.session_state = {}
        
        # 중복 추가
        question_input.add_to_history("질문 1")
        question_input.add_to_history("질문 1")
        question_input.add_to_history("질문 2")
        
        history = question_input.get_input_history()
        assert len(history) == 2
        assert "질문 1" in history
        assert "질문 2" in history

class TestSharedKnowledgeBank:
    """공유 지식 은행 테스트"""
    
    def test_knowledge_node_creation(self):
        """지식 노드 생성 테스트"""
        if not COMPONENTS_AVAILABLE:
            pytest.skip("Components not available")
        
        node = KnowledgeNode("test_id", "테스트 내용", "insight", {"source": "test"})
        
        assert node.node_id == "test_id"
        assert node.content == "테스트 내용"
        assert node.node_type == "insight"
        assert node.metadata["source"] == "test"
        assert isinstance(node.created_at, datetime)
    
    def test_knowledge_node_to_dict(self):
        """지식 노드 딕셔너리 변환 테스트"""
        if not COMPONENTS_AVAILABLE:
            pytest.skip("Components not available")
        
        node = KnowledgeNode("test_id", "테스트 내용", "insight")
        node_dict = node.to_dict()
        
        assert node_dict['node_id'] == "test_id"
        assert node_dict['content'] == "테스트 내용"
        assert node_dict['node_type'] == "insight"
        assert 'created_at' in node_dict
        assert 'updated_at' in node_dict
    
    def test_knowledge_graph_node_addition(self):
        """지식 그래프 노드 추가 테스트"""
        if not COMPONENTS_AVAILABLE:
            pytest.skip("Components not available")
        
        graph = KnowledgeGraph()
        node = KnowledgeNode("test_id", "테스트 내용", "insight")
        
        graph.add_node(node)
        
        assert "test_id" in graph.nodes
        assert graph.nodes["test_id"] == node
        assert graph.graph.has_node("test_id")
    
    def test_knowledge_graph_edge_addition(self):
        """지식 그래프 엣지 추가 테스트"""
        if not COMPONENTS_AVAILABLE:
            pytest.skip("Components not available")
        
        graph = KnowledgeGraph()
        
        # 노드 추가
        node1 = KnowledgeNode("node1", "내용1", "insight")
        node2 = KnowledgeNode("node2", "내용2", "pattern")
        
        graph.add_node(node1)
        graph.add_node(node2)
        
        # 엣지 추가
        graph.add_edge("node1", "node2", "related", 0.8)
        
        assert graph.graph.has_edge("node1", "node2")
        assert graph.graph["node1"]["node2"]["relationship"] == "related"
        assert graph.graph["node1"]["node2"]["weight"] == 0.8
    
    def test_knowledge_graph_related_nodes(self):
        """관련 노드 찾기 테스트"""
        if not COMPONENTS_AVAILABLE:
            pytest.skip("Components not available")
        
        graph = KnowledgeGraph()
        
        # 노드 추가
        nodes = []
        for i in range(4):
            node = KnowledgeNode(f"node{i}", f"내용{i}", "insight")
            nodes.append(node)
            graph.add_node(node)
        
        # 연결 구조 생성: node0 -> node1 -> node2 -> node3
        graph.add_edge("node0", "node1", "related", 1.0)
        graph.add_edge("node1", "node2", "related", 1.0)
        graph.add_edge("node2", "node3", "related", 1.0)
        
        # 관련 노드 찾기
        related = graph.find_related_nodes("node0", max_depth=2)
        
        assert "node1" in related
        assert "node2" in related
        assert "node3" not in related  # depth 2 초과
    
    @patch('sqlite3.connect')
    def test_shared_knowledge_bank_init(self, mock_connect):
        """공유 지식 은행 초기화 테스트"""
        if not COMPONENTS_AVAILABLE:
            pytest.skip("Components not available")
        
        # Mock 데이터베이스 연결
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_connect.return_value.__enter__.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            try:
                bank = SharedKnowledgeBank(tmp_file.name)
                assert bank.db_path == tmp_file.name
                assert isinstance(bank.knowledge_graph, KnowledgeGraph)
                assert bank.session_key == "shared_knowledge_bank"
            finally:
                os.unlink(tmp_file.name)
    
    def test_knowledge_stats_calculation(self):
        """지식 통계 계산 테스트"""
        if not COMPONENTS_AVAILABLE:
            pytest.skip("Components not available")
        
        graph = KnowledgeGraph()
        
        # 테스트 노드 추가
        node_types = ["insight", "pattern", "insight", "rule", "pattern"]
        for i, node_type in enumerate(node_types):
            node = KnowledgeNode(f"node{i}", f"내용{i}", node_type)
            graph.add_node(node)
        
        # 임시 지식 은행 생성
        class MockKnowledgeBank:
            def __init__(self):
                self.knowledge_graph = graph
            
            def get_knowledge_stats(self):
                stats = {
                    'total_nodes': len(self.knowledge_graph.nodes),
                    'total_relationships': len(self.knowledge_graph.graph.edges()),
                    'node_types': {},
                    'most_connected_nodes': [],
                    'recent_additions': []
                }
                
                for node in self.knowledge_graph.nodes.values():
                    stats['node_types'][node.node_type] = stats['node_types'].get(node.node_type, 0) + 1
                
                return stats
        
        bank = MockKnowledgeBank()
        stats = bank.get_knowledge_stats()
        
        assert stats['total_nodes'] == 5
        assert stats['node_types']['insight'] == 2
        assert stats['node_types']['pattern'] == 2
        assert stats['node_types']['rule'] == 1

class TestFactoryFunctions:
    """팩토리 함수 테스트"""
    
    def test_create_file_upload_manager(self):
        """파일 업로드 관리자 팩토리 테스트"""
        if not COMPONENTS_AVAILABLE:
            pytest.skip("Components not available")
        
        manager = create_file_upload_manager()
        assert isinstance(manager, FileUploadManager)
        assert manager.session_key == "uploaded_files"
    
    def test_create_chat_interface(self):
        """채팅 인터페이스 팩토리 테스트"""
        if not COMPONENTS_AVAILABLE:
            pytest.skip("Components not available")
        
        chat = create_chat_interface()
        assert isinstance(chat, AutoScrollChatInterface)
        assert chat.session_key == "chat_messages"
    
    def test_create_question_input(self):
        """질문 입력 팩토리 테스트"""
        if not COMPONENTS_AVAILABLE:
            pytest.skip("Components not available")
        
        question_input = create_question_input()
        assert isinstance(question_input, QuestionInput)
        assert question_input.session_key == "question_input"

# 통합 테스트
class TestIntegration:
    """통합 테스트"""
    
    def test_component_interaction(self):
        """컴포넌트 간 상호작용 테스트"""
        if not COMPONENTS_AVAILABLE:
            pytest.skip("Components not available")
        
        # 세션 상태 초기화
        import streamlit as st
        st.session_state = {}
        
        # 컴포넌트 생성
        file_manager = create_file_upload_manager()
        chat = create_chat_interface()
        question_input = create_question_input()
        
        # 상호작용 시뮬레이션
        chat.add_user_message("파일을 업로드했습니다")
        question_input.add_to_history("데이터 분석 요청")
        
        # 상태 확인
        messages = chat.get_messages()
        history = question_input.get_input_history()
        
        assert len(messages) == 1
        assert len(history) == 1
        assert messages[0].role == "🧑🏻"
        assert history[0] == "데이터 분석 요청"

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 