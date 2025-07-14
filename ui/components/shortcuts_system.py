#!/usr/bin/env python3
"""
⌨️ CherryAI 바로가기 시스템

ChatGPT/Claude 수준의 키보드 단축키 지원 시스템

Key Features:
- 키보드 단축키 지원 (Ctrl/Cmd 조합)
- 사용자 정의 바로가기
- 컨텍스트별 단축키
- 접근성 개선 (스크린 리더 지원)
- 바로가기 도움말
- 충돌 방지 및 우선순위 관리

Architecture:
- Shortcut Manager: 단축키 등록 및 관리
- Event Handler: 키보드 이벤트 처리
- Context Manager: 컨텍스트별 단축키 활성화
- Help System: 바로가기 도움말 제공
"""

import streamlit as st
import json
from typing import Dict, List, Optional, Any, Callable, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class ShortcutContext(Enum):
    """바로가기 컨텍스트"""
    GLOBAL = "global"  # 전역 단축키
    CHAT = "chat"  # 채팅 영역
    EDITOR = "editor"  # 편집기 영역
    FILE_UPLOAD = "file_upload"  # 파일 업로드
    SESSION = "session"  # 세션 관리
    NAVIGATION = "navigation"  # 네비게이션

class ModifierKey(Enum):
    """수정 키"""
    CTRL = "ctrl"
    ALT = "alt"
    SHIFT = "shift"
    META = "meta"  # Cmd on Mac, Win on Windows

@dataclass
class Shortcut:
    """바로가기 정의"""
    id: str
    name: str
    description: str
    key: str
    modifiers: List[ModifierKey]
    context: ShortcutContext
    action: str  # JavaScript 함수명 또는 액션 ID
    enabled: bool = True
    custom: bool = False  # 사용자 정의 여부
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "key": self.key,
            "modifiers": [m.value for m in self.modifiers],
            "context": self.context.value,
            "action": self.action,
            "enabled": self.enabled,
            "custom": self.custom
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Shortcut':
        """딕셔너리에서 생성"""
        return cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            key=data["key"],
            modifiers=[ModifierKey(m) for m in data["modifiers"]],
            context=ShortcutContext(data["context"]),
            action=data["action"],
            enabled=data["enabled"],
            custom=data["custom"]
        )
    
    def get_display_text(self) -> str:
        """표시용 텍스트 생성"""
        modifier_texts = []
        for modifier in self.modifiers:
            if modifier == ModifierKey.CTRL:
                modifier_texts.append("Ctrl")
            elif modifier == ModifierKey.ALT:
                modifier_texts.append("Alt")
            elif modifier == ModifierKey.SHIFT:
                modifier_texts.append("Shift")
            elif modifier == ModifierKey.META:
                modifier_texts.append("Cmd")  # Mac 기준으로 표시
        
        if modifier_texts:
            return f"{'+'.join(modifier_texts)}+{self.key.upper()}"
        else:
            return self.key.upper()

class ShortcutsManager:
    """
    ⌨️ 바로가기 관리자
    
    모든 키보드 단축키를 관리하고 처리
    """
    
    def __init__(self):
        """바로가기 관리자 초기화"""
        self.shortcuts: Dict[str, Shortcut] = {}
        self.context_shortcuts: Dict[ShortcutContext, List[str]] = {}
        self.active_contexts: Set[ShortcutContext] = {ShortcutContext.GLOBAL}
        
        # 기본 바로가기 등록
        self._register_default_shortcuts()
        
        # 사용자 정의 바로가기 로드
        self._load_custom_shortcuts()
        
        logger.info("⌨️ 바로가기 관리자 초기화 완료")
    
    def _register_default_shortcuts(self) -> None:
        """기본 바로가기 등록"""
        
        # 전역 바로가기
        self.register_shortcut(Shortcut(
            id="new_session",
            name="새 세션",
            description="새로운 채팅 세션을 시작합니다",
            key="n",
            modifiers=[ModifierKey.CTRL],
            context=ShortcutContext.GLOBAL,
            action="createNewSession"
        ))
        
        self.register_shortcut(Shortcut(
            id="save_session",
            name="세션 저장",
            description="현재 세션을 저장합니다",
            key="s",
            modifiers=[ModifierKey.CTRL],
            context=ShortcutContext.GLOBAL,
            action="saveCurrentSession"
        ))
        
        self.register_shortcut(Shortcut(
            id="open_session",
            name="세션 열기",
            description="세션 목록을 열어 선택합니다",
            key="o",
            modifiers=[ModifierKey.CTRL],
            context=ShortcutContext.GLOBAL,
            action="openSessionList"
        ))
        
        self.register_shortcut(Shortcut(
            id="toggle_sidebar",
            name="사이드바 토글",
            description="왼쪽 사이드바를 열거나 닫습니다",
            key="b",
            modifiers=[ModifierKey.CTRL],
            context=ShortcutContext.GLOBAL,
            action="toggleSidebar"
        ))
        
        self.register_shortcut(Shortcut(
            id="focus_input",
            name="입력창 포커스",
            description="채팅 입력창에 포커스를 이동합니다",
            key="/",
            modifiers=[],
            context=ShortcutContext.GLOBAL,
            action="focusChatInput"
        ))
        
        self.register_shortcut(Shortcut(
            id="search_sessions",
            name="세션 검색",
            description="세션 검색창을 엽니다",
            key="f",
            modifiers=[ModifierKey.CTRL],
            context=ShortcutContext.GLOBAL,
            action="openSessionSearch"
        ))
        
        # 채팅 영역 바로가기
        self.register_shortcut(Shortcut(
            id="send_message",
            name="메시지 전송",
            description="현재 입력된 메시지를 전송합니다",
            key="Enter",
            modifiers=[],
            context=ShortcutContext.CHAT,
            action="sendMessage"
        ))
        
        self.register_shortcut(Shortcut(
            id="new_line",
            name="줄바꿈",
            description="새 줄을 추가합니다",
            key="Enter",
            modifiers=[ModifierKey.SHIFT],
            context=ShortcutContext.CHAT,
            action="addNewLine"
        ))
        
        self.register_shortcut(Shortcut(
            id="clear_input",
            name="입력 지우기",
            description="채팅 입력창을 지웁니다",
            key="l",
            modifiers=[ModifierKey.CTRL],
            context=ShortcutContext.CHAT,
            action="clearChatInput"
        ))
        
        self.register_shortcut(Shortcut(
            id="upload_file",
            name="파일 업로드",
            description="파일 업로드 창을 엽니다",
            key="u",
            modifiers=[ModifierKey.CTRL],
            context=ShortcutContext.CHAT,
            action="openFileUpload"
        ))
        
        # 파일 업로드 바로가기
        self.register_shortcut(Shortcut(
            id="select_files",
            name="파일 선택",
            description="업로드할 파일을 선택합니다",
            key="o",
            modifiers=[ModifierKey.CTRL],
            context=ShortcutContext.FILE_UPLOAD,
            action="selectFiles"
        ))
        
        # 세션 관리 바로가기
        self.register_shortcut(Shortcut(
            id="delete_session",
            name="세션 삭제",
            description="현재 세션을 삭제합니다",
            key="Delete",
            modifiers=[],
            context=ShortcutContext.SESSION,
            action="deleteCurrentSession"
        ))
        
        self.register_shortcut(Shortcut(
            id="favorite_session",
            name="즐겨찾기 토글",
            description="현재 세션의 즐겨찾기를 토글합니다",
            key="d",
            modifiers=[ModifierKey.CTRL],
            context=ShortcutContext.SESSION,
            action="toggleSessionFavorite"
        ))
        
        # 네비게이션 바로가기
        self.register_shortcut(Shortcut(
            id="previous_session",
            name="이전 세션",
            description="이전 세션으로 이동합니다",
            key="ArrowUp",
            modifiers=[ModifierKey.CTRL],
            context=ShortcutContext.NAVIGATION,
            action="goToPreviousSession"
        ))
        
        self.register_shortcut(Shortcut(
            id="next_session",
            name="다음 세션",
            description="다음 세션으로 이동합니다",
            key="ArrowDown",
            modifiers=[ModifierKey.CTRL],
            context=ShortcutContext.NAVIGATION,
            action="goToNextSession"
        ))
        
        # 도움말 바로가기
        self.register_shortcut(Shortcut(
            id="show_help",
            name="도움말",
            description="바로가기 도움말을 표시합니다",
            key="?",
            modifiers=[ModifierKey.CTRL],
            context=ShortcutContext.GLOBAL,
            action="showShortcutHelp"
        ))
    
    def register_shortcut(self, shortcut: Shortcut) -> bool:
        """바로가기 등록"""
        try:
            # 중복 검사
            if self._check_conflict(shortcut):
                logger.warning(f"바로가기 충돌: {shortcut.id}")
                return False
            
            self.shortcuts[shortcut.id] = shortcut
            
            # 컨텍스트별 인덱스 업데이트
            if shortcut.context not in self.context_shortcuts:
                self.context_shortcuts[shortcut.context] = []
            self.context_shortcuts[shortcut.context].append(shortcut.id)
            
            logger.info(f"⌨️ 바로가기 등록됨: {shortcut.id} ({shortcut.get_display_text()})")
            return True
            
        except Exception as e:
            logger.error(f"바로가기 등록 실패: {shortcut.id} - {e}")
            return False
    
    def unregister_shortcut(self, shortcut_id: str) -> bool:
        """바로가기 등록 해제"""
        try:
            if shortcut_id not in self.shortcuts:
                return False
            
            shortcut = self.shortcuts[shortcut_id]
            del self.shortcuts[shortcut_id]
            
            # 컨텍스트별 인덱스에서 제거
            if shortcut.context in self.context_shortcuts:
                if shortcut_id in self.context_shortcuts[shortcut.context]:
                    self.context_shortcuts[shortcut.context].remove(shortcut_id)
            
            logger.info(f"⌨️ 바로가기 해제됨: {shortcut_id}")
            return True
            
        except Exception as e:
            logger.error(f"바로가기 해제 실패: {shortcut_id} - {e}")
            return False
    
    def _check_conflict(self, new_shortcut: Shortcut) -> bool:
        """바로가기 충돌 검사"""
        for existing_shortcut in self.shortcuts.values():
            if (existing_shortcut.key == new_shortcut.key and
                existing_shortcut.modifiers == new_shortcut.modifiers and
                existing_shortcut.context == new_shortcut.context and
                existing_shortcut.enabled):
                return True
        return False
    
    def get_shortcuts_by_context(self, context: ShortcutContext) -> List[Shortcut]:
        """컨텍스트별 바로가기 조회"""
        shortcut_ids = self.context_shortcuts.get(context, [])
        return [self.shortcuts[sid] for sid in shortcut_ids 
                if sid in self.shortcuts and self.shortcuts[sid].enabled]
    
    def get_all_shortcuts(self) -> List[Shortcut]:
        """모든 바로가기 조회"""
        return [s for s in self.shortcuts.values() if s.enabled]
    
    def set_active_contexts(self, contexts: Set[ShortcutContext]) -> None:
        """활성 컨텍스트 설정"""
        self.active_contexts = contexts
        # 전역 컨텍스트는 항상 활성
        self.active_contexts.add(ShortcutContext.GLOBAL)
    
    def add_active_context(self, context: ShortcutContext) -> None:
        """활성 컨텍스트 추가"""
        self.active_contexts.add(context)
    
    def remove_active_context(self, context: ShortcutContext) -> None:
        """활성 컨텍스트 제거"""
        if context != ShortcutContext.GLOBAL:  # 전역 컨텍스트는 제거 불가
            self.active_contexts.discard(context)
    
    def get_active_shortcuts(self) -> List[Shortcut]:
        """현재 활성 컨텍스트의 바로가기들 조회"""
        active_shortcuts = []
        for context in self.active_contexts:
            active_shortcuts.extend(self.get_shortcuts_by_context(context))
        return active_shortcuts
    
    def enable_shortcut(self, shortcut_id: str) -> bool:
        """바로가기 활성화"""
        if shortcut_id in self.shortcuts:
            self.shortcuts[shortcut_id].enabled = True
            return True
        return False
    
    def disable_shortcut(self, shortcut_id: str) -> bool:
        """바로가기 비활성화"""
        if shortcut_id in self.shortcuts:
            self.shortcuts[shortcut_id].enabled = False
            return True
        return False
    
    def _load_custom_shortcuts(self) -> None:
        """사용자 정의 바로가기 로드"""
        try:
            # Streamlit session state에서 로드
            if "custom_shortcuts" in st.session_state:
                custom_shortcuts_data = st.session_state["custom_shortcuts"]
                for shortcut_data in custom_shortcuts_data:
                    shortcut = Shortcut.from_dict(shortcut_data)
                    self.register_shortcut(shortcut)
                logger.info("⌨️ 사용자 정의 바로가기 로드됨")
        except Exception as e:
            logger.error(f"사용자 정의 바로가기 로드 실패: {e}")
    
    def save_custom_shortcuts(self) -> bool:
        """사용자 정의 바로가기 저장"""
        try:
            custom_shortcuts = [s.to_dict() for s in self.shortcuts.values() if s.custom]
            st.session_state["custom_shortcuts"] = custom_shortcuts
            logger.info("⌨️ 사용자 정의 바로가기 저장됨")
            return True
        except Exception as e:
            logger.error(f"사용자 정의 바로가기 저장 실패: {e}")
            return False
    
    def render_shortcuts_javascript(self) -> str:
        """바로가기용 JavaScript 코드 생성"""
        active_shortcuts = self.get_active_shortcuts()
        
        js_shortcuts = []
        for shortcut in active_shortcuts:
            # 수정 키 조합 생성
            modifiers_check = []
            for modifier in shortcut.modifiers:
                if modifier == ModifierKey.CTRL:
                    modifiers_check.append("event.ctrlKey")
                elif modifier == ModifierKey.ALT:
                    modifiers_check.append("event.altKey")
                elif modifier == ModifierKey.SHIFT:
                    modifiers_check.append("event.shiftKey")
                elif modifier == ModifierKey.META:
                    modifiers_check.append("event.metaKey")
            
            # 키 조건
            key_check = f"event.key === '{shortcut.key}'"
            if shortcut.key == "Enter":
                key_check = "event.key === 'Enter'"
            elif shortcut.key == "Delete":
                key_check = "event.key === 'Delete'"
            elif shortcut.key.startswith("Arrow"):
                key_check = f"event.key === '{shortcut.key}'"
            
            # 전체 조건
            conditions = [key_check] + modifiers_check
            condition_str = " && ".join(conditions)
            
            js_shortcut = f"""
            if ({condition_str}) {{
                event.preventDefault();
                {shortcut.action}();
                return false;
            }}
            """
            js_shortcuts.append(js_shortcut)
        
        return f"""
        function handleKeyboardShortcuts(event) {{
            {chr(10).join(js_shortcuts)}
        }}
        
        // 키보드 이벤트 리스너 등록
        document.addEventListener('keydown', handleKeyboardShortcuts);
        """
    
    def render_help_modal(self) -> None:
        """바로가기 도움말 모달 렌더링"""
        # 컨텍스트별로 그룹화
        context_groups = {}
        for shortcut in self.get_all_shortcuts():
            if shortcut.context not in context_groups:
                context_groups[shortcut.context] = []
            context_groups[shortcut.context].append(shortcut)
        
        # 모달 스타일
        st.markdown("""
        <style>
        .shortcut-help-container {
            background: white;
            border-radius: 12px;
            padding: 24px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            max-height: 80vh;
            overflow-y: auto;
        }
        
        .shortcut-group {
            margin-bottom: 24px;
        }
        
        .shortcut-group-title {
            font-size: 18px;
            font-weight: 600;
            color: #2d3748;
            margin-bottom: 12px;
            border-bottom: 2px solid #e2e8f0;
            padding-bottom: 8px;
        }
        
        .shortcut-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 8px 0;
            border-bottom: 1px solid #f1f3f4;
        }
        
        .shortcut-description {
            flex: 1;
            margin-right: 16px;
        }
        
        .shortcut-name {
            font-weight: 500;
            color: #2d3748;
        }
        
        .shortcut-desc {
            font-size: 14px;
            color: #718096;
            margin-top: 2px;
        }
        
        .shortcut-keys {
            background: #f7fafc;
            border: 1px solid #e2e8f0;
            border-radius: 6px;
            padding: 4px 8px;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 12px;
            color: #4a5568;
            white-space: nowrap;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # 컨텍스트별 바로가기 표시
        context_names = {
            ShortcutContext.GLOBAL: "🌍 전역",
            ShortcutContext.CHAT: "💬 채팅",
            ShortcutContext.EDITOR: "✏️ 편집기",
            ShortcutContext.FILE_UPLOAD: "📁 파일 업로드",
            ShortcutContext.SESSION: "📚 세션 관리",
            ShortcutContext.NAVIGATION: "🧭 네비게이션"
        }
        
        help_content = '<div class="shortcut-help-container">'
        help_content += '<h2 style="text-align: center; margin-bottom: 24px;">⌨️ 키보드 바로가기</h2>'
        
        for context, shortcuts in context_groups.items():
            if not shortcuts:
                continue
                
            context_name = context_names.get(context, context.value.title())
            help_content += f'<div class="shortcut-group">'
            help_content += f'<div class="shortcut-group-title">{context_name}</div>'
            
            for shortcut in sorted(shortcuts, key=lambda x: x.name):
                help_content += f'''
                <div class="shortcut-item">
                    <div class="shortcut-description">
                        <div class="shortcut-name">{shortcut.name}</div>
                        <div class="shortcut-desc">{shortcut.description}</div>
                    </div>
                    <div class="shortcut-keys">{shortcut.get_display_text()}</div>
                </div>
                '''
            
            help_content += '</div>'
        
        help_content += '</div>'
        
        st.markdown(help_content, unsafe_allow_html=True)

# Streamlit 컴포넌트 함수들
def inject_shortcuts_javascript(shortcuts_manager: ShortcutsManager):
    """바로가기 JavaScript 주입"""
    js_code = shortcuts_manager.render_shortcuts_javascript()
    
    # 액션 함수들 정의
    action_functions = """
    // 바로가기 액션 함수들
    function createNewSession() {
        console.log('새 세션 생성');
        // Streamlit 이벤트 트리거
        window.parent.postMessage({type: 'streamlit:shortcut', action: 'new_session'}, '*');
    }
    
    function saveCurrentSession() {
        console.log('세션 저장');
        window.parent.postMessage({type: 'streamlit:shortcut', action: 'save_session'}, '*');
    }
    
    function openSessionList() {
        console.log('세션 목록 열기');
        window.parent.postMessage({type: 'streamlit:shortcut', action: 'open_session'}, '*');
    }
    
    function toggleSidebar() {
        console.log('사이드바 토글');
        window.parent.postMessage({type: 'streamlit:shortcut', action: 'toggle_sidebar'}, '*');
    }
    
    function focusChatInput() {
        const chatInput = document.querySelector('textarea[data-testid="stChatInput"]');
        if (chatInput) {
            chatInput.focus();
        }
    }
    
    function openSessionSearch() {
        console.log('세션 검색 열기');
        window.parent.postMessage({type: 'streamlit:shortcut', action: 'search_sessions'}, '*');
    }
    
    function sendMessage() {
        const submitButton = document.querySelector('[data-testid="stChatInputSubmitButton"]');
        if (submitButton) {
            submitButton.click();
        }
    }
    
    function addNewLine() {
        const chatInput = document.querySelector('textarea[data-testid="stChatInput"]');
        if (chatInput) {
            const cursorPosition = chatInput.selectionStart;
            const value = chatInput.value;
            chatInput.value = value.slice(0, cursorPosition) + '\\n' + value.slice(cursorPosition);
            chatInput.selectionStart = chatInput.selectionEnd = cursorPosition + 1;
        }
    }
    
    function clearChatInput() {
        const chatInput = document.querySelector('textarea[data-testid="stChatInput"]');
        if (chatInput) {
            chatInput.value = '';
            chatInput.focus();
        }
    }
    
    function openFileUpload() {
        console.log('파일 업로드 열기');
        window.parent.postMessage({type: 'streamlit:shortcut', action: 'upload_file'}, '*');
    }
    
    function selectFiles() {
        const fileInput = document.querySelector('input[type="file"]');
        if (fileInput) {
            fileInput.click();
        }
    }
    
    function deleteCurrentSession() {
        console.log('현재 세션 삭제');
        window.parent.postMessage({type: 'streamlit:shortcut', action: 'delete_session'}, '*');
    }
    
    function toggleSessionFavorite() {
        console.log('즐겨찾기 토글');
        window.parent.postMessage({type: 'streamlit:shortcut', action: 'favorite_session'}, '*');
    }
    
    function goToPreviousSession() {
        console.log('이전 세션으로 이동');
        window.parent.postMessage({type: 'streamlit:shortcut', action: 'previous_session'}, '*');
    }
    
    function goToNextSession() {
        console.log('다음 세션으로 이동');
        window.parent.postMessage({type: 'streamlit:shortcut', action: 'next_session'}, '*');
    }
    
    function showShortcutHelp() {
        console.log('바로가기 도움말 표시');
        window.parent.postMessage({type: 'streamlit:shortcut', action: 'show_help'}, '*');
    }
    """
    
    # 전체 JavaScript 코드
    full_js_code = f"""
    <script>
    {action_functions}
    
    {js_code}
    
    // Streamlit과의 통신을 위한 메시지 리스너
    window.addEventListener('message', function(event) {{
        if (event.data.type === 'streamlit:shortcut') {{
            console.log('바로가기 액션 실행:', event.data.action);
        }}
    }});
    </script>
    """
    
    st.markdown(full_js_code, unsafe_allow_html=True)

def render_shortcuts_info():
    """바로가기 정보 표시"""
    st.markdown("""
    <div style="
        background: #f8f9fa;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
        font-size: 12px;
        color: #6c757d;
        text-align: center;
    ">
        💡 <strong>Ctrl+?</strong>를 눌러 바로가기 도움말을 확인하세요
    </div>
    """, unsafe_allow_html=True)

# 전역 인스턴스 관리
_shortcuts_manager_instance = None

def get_shortcuts_manager() -> ShortcutsManager:
    """바로가기 관리자 싱글톤 인스턴스 반환"""
    global _shortcuts_manager_instance
    if _shortcuts_manager_instance is None:
        _shortcuts_manager_instance = ShortcutsManager()
    return _shortcuts_manager_instance

def initialize_shortcuts_manager() -> ShortcutsManager:
    """바로가기 관리자 초기화"""
    global _shortcuts_manager_instance
    _shortcuts_manager_instance = ShortcutsManager()
    inject_shortcuts_javascript(_shortcuts_manager_instance)
    return _shortcuts_manager_instance 