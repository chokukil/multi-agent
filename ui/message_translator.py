#!/usr/bin/env python3
"""
A2A 프로토콜 메시지를 사용자 친화적인 메시지로 변환하는 시스템
기술적인 내용을 자연스러운 언어로 번역하여 사용자 경험을 개선
"""

import re
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import streamlit as st

class MessageTranslator:
    """A2A 메시지를 사용자 친화적으로 변환하는 클래스"""
    
    def __init__(self):
        # 에이전트 이름 한국어 매핑
        self.agent_names = {
            'pandas_data_analyst': '📊 데이터 분석가',
            'data_validator': '🔍 데이터 검증가',
            'eda_analyst': '📈 탐색적 분석가',
            'visualization_expert': '📊 시각화 전문가',
            'ml_specialist': '🤖 머신러닝 전문가',
            'statistical_analyst': '📐 통계 분석가',
            'report_generator': '📝 보고서 작성가'
        }
        
        # 기술 용어 번역
        self.tech_translations = {
            'messageId': '메시지 ID',
            'response_type': '응답 유형',
            'direct_message': '직접 메시지',
            'task_response': '작업 응답',
            'ValidationError': '검증 오류',
            'HTTP Error 503': '서버 연결 오류',
            'timeout': '시간 초과',
            'connection_failed': '연결 실패'
        }
        
        # 상태 메시지 템플릿
        self.status_templates = {
            'thinking': [
                "🤔 {agent}가 문제를 분석하고 있습니다...",
                "💭 {agent}가 최적의 해결책을 찾고 있습니다...",
                "🔍 {agent}가 데이터를 면밀히 검토 중입니다..."
            ],
            'processing': [
                "⚙️ {agent}가 데이터를 처리하고 있습니다...",
                "🔄 {agent}가 분석을 수행 중입니다...",
                "📊 {agent}가 결과를 생성하고 있습니다..."
            ],
            'completed': [
                "✅ {agent}가 작업을 완료했습니다!",
                "🎉 {agent}의 분석이 성공적으로 끝났습니다!",
                "💫 {agent}가 훌륭한 결과를 만들어냈습니다!"
            ],
            'error': [
                "⚠️ {agent}가 문제를 발견했습니다.",
                "🔧 {agent}가 해결책을 찾고 있습니다.",
                "💡 {agent}가 대안을 제시합니다."
            ]
        }
    
    def translate_a2a_message(self, raw_message: Dict[str, Any]) -> Dict[str, Any]:
        """A2A 원시 메시지를 사용자 친화적 메시지로 변환"""
        
        # 메시지 ID와 기본 정보 추출
        message_id = raw_message.get('messageId', 'unknown')
        parts = raw_message.get('parts', [])
        response_type = raw_message.get('response_type', 'unknown')
        
        # 메시지 내용 추출
        content = self._extract_content_from_parts(parts)
        
        # 에이전트 정보 추출
        agent_info = self._identify_agent_from_message(content)
        
        # 메시지 유형 판단
        message_type = self._classify_message_type(content, response_type)
        
        # 사용자 친화적 메시지 생성
        friendly_message = self._create_friendly_message(
            content, agent_info, message_type, message_id
        )
        
        return {
            'original_message_id': message_id,
            'agent_name': agent_info['display_name'],
            'agent_icon': agent_info['icon'],
            'message_type': message_type,
            'friendly_content': friendly_message,
            'raw_content': content,
            'timestamp': datetime.now().strftime("%H:%M:%S"),
            'show_raw': False  # 기본적으로 원시 메시지 숨김
        }
    
    def _extract_content_from_parts(self, parts: List[Any]) -> str:
        """메시지 파트에서 실제 내용 추출"""
        if not parts:
            return ""
        
        content_parts = []
        for part in parts:
            if hasattr(part, 'root') and hasattr(part.root, 'text'):
                content_parts.append(part.root.text)
            elif isinstance(part, dict) and 'text' in part:
                content_parts.append(part['text'])
            elif isinstance(part, str):
                content_parts.append(part)
        
        return '\n'.join(content_parts)
    
    def _identify_agent_from_message(self, content: str) -> Dict[str, str]:
        """메시지 내용에서 에이전트 정보 추출"""
        # 기본값
        default_agent = {
            'name': 'unknown_agent',
            'display_name': '🤖 AI 어시스턴트',
            'icon': '🤖'
        }
        
        # 메시지에서 에이전트 단서 찾기
        content_lower = content.lower()
        
        for agent_key, display_name in self.agent_names.items():
            if agent_key in content_lower or any(keyword in content_lower for keyword in [
                'pandas', 'dataframe', '데이터프레임', '분석', 'analysis'
            ]):
                return {
                    'name': agent_key,
                    'display_name': display_name,
                    'icon': display_name.split()[0]  # 첫 번째 이모지 추출
                }
        
        return default_agent
    
    def _classify_message_type(self, content: str, response_type: str) -> str:
        """메시지 유형 분류"""
        content_lower = content.lower()
        
        # 오류 메시지 감지
        if any(error_keyword in content_lower for error_keyword in [
            'error', '오류', 'failed', '실패', 'not found', '찾을 수 없', 'dataset not found'
        ]):
            return 'error'
        
        # 성공 메시지 감지
        if any(success_keyword in content_lower for success_keyword in [
            'success', '성공', 'completed', '완료', 'analysis results', '분석 결과'
        ]):
            return 'success'
        
        # 진행 중 메시지 감지
        if any(progress_keyword in content_lower for progress_keyword in [
            'processing', '처리 중', 'analyzing', '분석 중', 'working', '작업 중'
        ]):
            return 'processing'
        
        # 정보 메시지 감지
        if any(info_keyword in content_lower for info_keyword in [
            'available', '사용 가능', 'dataset', '데이터셋', 'information', '정보'
        ]):
            return 'info'
        
        return 'general'
    
    def _create_friendly_message(self, content: str, agent_info: Dict[str, str], 
                                message_type: str, message_id: str) -> str:
        """사용자 친화적 메시지 생성"""
        
        agent_name = agent_info['display_name']
        
        if message_type == 'error':
            return self._create_error_message(content, agent_name)
        elif message_type == 'success':
            return self._create_success_message(content, agent_name)
        elif message_type == 'processing':
            return self._create_processing_message(content, agent_name)
        elif message_type == 'info':
            return self._create_info_message(content, agent_name)
        else:
            return self._create_general_message(content, agent_name)
    
    def _create_error_message(self, content: str, agent_name: str) -> str:
        """오류 메시지를 친화적으로 변환"""
        
        # 데이터셋을 찾을 수 없는 경우
        if 'Dataset Not Found' in content or 'dataset not found' in content.lower():
            # 사용 가능한 데이터셋 추출
            available_datasets = self._extract_available_datasets(content)
            
            message = f"""
            🔍 **{agent_name}의 알림**
            
            요청하신 데이터셋을 찾을 수 없었습니다.
            
            """
            
            if available_datasets:
                message += f"""
                📋 **현재 사용 가능한 데이터셋:**
                {chr(10).join([f'• `{dataset}`' for dataset in available_datasets])}
                
                💡 **해결 방법:**
                - 위 데이터셋 중 하나를 선택해 주세요
                - 또는 Data Loader 페이지에서 새 데이터를 업로드하세요
                """
            else:
                message += """
                📂 **현재 로드된 데이터가 없습니다.**
                
                💡 **해결 방법:**
                - Data Loader 페이지에서 데이터를 업로드해 주세요
                - CSV, Excel 파일을 지원합니다
                """
            
            return message.strip()
        
        # 일반적인 오류 메시지
        return f"""
        ⚠️ **{agent_name}의 알림**
        
        작업을 수행하는 중에 문제가 발생했습니다.
        
        🔧 **문제 해결을 위해 다음을 확인해 주세요:**
        - 데이터가 올바르게 로드되었는지 확인
        - 요청 내용이 명확한지 확인
        - 잠시 후 다시 시도
        
        💬 **상세 정보가 필요하시면 '원시 메시지 보기'를 클릭하세요.**
        """
    
    def _create_success_message(self, content: str, agent_name: str) -> str:
        """성공 메시지를 친화적으로 변환"""
        
        # 분석 결과인 경우
        if '# 📊 Data Analysis Results' in content or 'analysis results' in content.lower():
            return f"""
            🎉 **{agent_name}가 분석을 완료했습니다!**
            
            📊 **분석 결과가 준비되었습니다:**
            - 데이터 구조 분석 완료
            - 통계적 요약 생성
            - 주요 인사이트 도출
            - 시각화 권장사항 제시
            
            ✨ **아래에서 상세한 분석 결과를 확인하세요!**
            """
        
        return f"""
        ✅ **{agent_name}의 작업 완료**
        
        요청하신 작업이 성공적으로 완료되었습니다.
        결과를 아래에서 확인해 주세요!
        """
    
    def _create_processing_message(self, content: str, agent_name: str) -> str:
        """처리 중 메시지를 친화적으로 변환"""
        return f"""
        ⚙️ **{agent_name}가 작업 중입니다...**
        
        현재 데이터를 분석하고 있습니다. 잠시만 기다려 주세요.
        
        🔄 **진행 상황:**
        - 데이터 로딩 및 검증
        - 분석 알고리즘 적용
        - 결과 정리 및 포맷팅
        """
    
    def _create_info_message(self, content: str, agent_name: str) -> str:
        """정보 메시지를 친화적으로 변환"""
        return f"""
        💡 **{agent_name}의 정보**
        
        {self._clean_technical_content(content)}
        """
    
    def _create_general_message(self, content: str, agent_name: str) -> str:
        """일반 메시지를 친화적으로 변환"""
        cleaned_content = self._clean_technical_content(content)
        
        return f"""
        💬 **{agent_name}의 메시지**
        
        {cleaned_content}
        """
    
    def _extract_available_datasets(self, content: str) -> List[str]:
        """메시지에서 사용 가능한 데이터셋 목록 추출"""
        datasets = []
        
        # "Available datasets:" 다음에 오는 목록 찾기
        lines = content.split('\n')
        in_dataset_section = False
        
        for line in lines:
            line = line.strip()
            
            if 'available datasets' in line.lower():
                in_dataset_section = True
                continue
            
            if in_dataset_section:
                # • 또는 - 로 시작하는 라인에서 데이터셋 이름 추출
                if line.startswith('•') or line.startswith('-'):
                    # 백틱으로 감싸진 데이터셋 이름 추출
                    match = re.search(r'`([^`]+)`', line)
                    if match:
                        datasets.append(match.group(1))
                elif line == '' or line.startswith('**'):
                    # 빈 줄이나 새로운 섹션이 시작되면 종료
                    break
        
        return datasets
    
    def _clean_technical_content(self, content: str) -> str:
        """기술적 내용을 사용자 친화적으로 정리"""
        
        # 기술 용어 번역
        cleaned = content
        for tech_term, translation in self.tech_translations.items():
            cleaned = cleaned.replace(tech_term, translation)
        
        # JSON 구조 제거
        cleaned = re.sub(r'\{[^}]*\}', '', cleaned)
        
        # 메시지 ID 패턴 제거
        cleaned = re.sub(r'[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}', 
                        '[메시지 ID]', cleaned)
        
        # 불필요한 공백 정리
        cleaned = re.sub(r'\n\s*\n', '\n\n', cleaned)
        cleaned = cleaned.strip()
        
        return cleaned


class MessageRenderer:
    """변환된 메시지를 아름답게 렌더링하는 클래스"""
    
    def __init__(self):
        self.translator = MessageTranslator()
    
    def render_a2a_message(self, raw_message: Dict[str, Any], 
                          container: Optional[st.container] = None) -> None:
        """A2A 메시지를 아름답게 렌더링"""
        
        if container is None:
            container = st.container()
        
        with container:
            # 메시지 번역
            translated = self.translator.translate_a2a_message(raw_message)
            
            # 메시지 타입에 따른 스타일링
            self._render_message_card(translated)
    
    def _render_message_card(self, message: Dict[str, Any]) -> None:
        """메시지를 카드 형태로 렌더링"""
        
        agent_name = message['agent_name']
        agent_icon = message['agent_icon']
        message_type = message['message_type']
        friendly_content = message['friendly_content']
        timestamp = message['timestamp']
        
        # 메시지 타입별 색상 설정
        colors = {
            'error': '#e74c3c',
            'success': '#2ecc71',
            'processing': '#3498db',
            'info': '#f39c12',
            'general': '#95a5a6'
        }
        
        color = colors.get(message_type, colors['general'])
        
        # 메시지 카드 HTML
        card_html = f"""
        <div style="
            background: linear-gradient(135deg, {color}15 0%, {color}05 100%);
            border-left: 4px solid {color};
            border-radius: 12px;
            padding: 20px;
            margin: 15px 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        ">
            <div style="
                display: flex;
                align-items: center;
                margin-bottom: 15px;
                padding-bottom: 10px;
                border-bottom: 1px solid {color}30;
            ">
                <div style="
                    font-size: 24px;
                    margin-right: 12px;
                ">
                    {agent_icon}
                </div>
                <div style="flex-grow: 1;">
                    <h4 style="
                        margin: 0;
                        color: #2c3e50;
                        font-size: 16px;
                        font-weight: 600;
                    ">
                        {agent_name}
                    </h4>
                </div>
                <div style="
                    font-size: 12px;
                    color: #7f8c8d;
                    background: rgba(255,255,255,0.8);
                    padding: 4px 8px;
                    border-radius: 4px;
                ">
                    {timestamp}
                </div>
            </div>
            
            <div style="
                color: #2c3e50;
                line-height: 1.6;
                font-size: 14px;
            ">
                {friendly_content.replace(chr(10), '<br>')}
            </div>
        </div>
        """
        
        st.markdown(card_html, unsafe_allow_html=True)
        
        # 원시 메시지 보기 옵션
        with st.expander("🔧 원시 메시지 보기 (개발자용)", expanded=False):
            st.code(message['raw_content'], language='text')


# 사용 예시 및 테스트 함수
def demo_message_translation():
    """메시지 번역 데모"""
    st.title("🔄 메시지 번역 시연")
    
    # 샘플 A2A 메시지들
    sample_messages = [
        {
            "messageId": "d5382743-49e1-4938-8f92-28921f14ca2f",
            "parts": [
                {
                    "root": {
                        "text": "❌ **Dataset Not Found: 'titanic.csv'**\n\n**Available datasets:**\n• `sample_sales_data.csv`\n\n**Solution:** Use one of the available dataset IDs above, or upload new data via the Data Loader page."
                    }
                }
            ],
            "response_type": "direct_message"
        },
        {
            "messageId": "72620c50-ebeb-4269-9a45-dbfa74b5b5c0",
            "parts": [
                {
                    "root": {
                        "text": "# 📊 Data Analysis Results for `titanic.csv`\n\nOkay, here's an analysis of the Titanic dataset..."
                    }
                }
            ],
            "response_type": "direct_message"
        }
    ]
    
    renderer = MessageRenderer()
    
    st.subheader("변환 전 (기술적 메시지)")
    for i, msg in enumerate(sample_messages):
        with st.expander(f"원시 메시지 {i+1}", expanded=False):
            st.json(msg)
    
    st.subheader("변환 후 (사용자 친화적 메시지)")
    for msg in sample_messages:
        renderer.render_a2a_message(msg)

if __name__ == "__main__":
    demo_message_translation() 