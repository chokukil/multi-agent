"""
Enhanced Exception Handling System

검증된 Universal Engine 패턴:
- SpecificExceptionTypes: 도메인별 예외 타입
- IntelligentErrorRecovery: 지능형 오류 복구
- ContextualErrorLogging: 문맥 인식 로깅
- UserFriendlyErrorMessages: 사용자 친화적 메시지
"""

import logging
import traceback
import asyncio
from typing import Dict, Any, Optional, Callable, List, Type
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import streamlit as st

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """오류 심각도"""
    LOW = "low"          # 사용자에게 알림 불필요
    MEDIUM = "medium"    # 사용자에게 경고 표시
    HIGH = "high"        # 사용자에게 명확한 오류 메시지
    CRITICAL = "critical"  # 시스템 복구 필요


class ErrorCategory(Enum):
    """오류 카테고리"""
    FILE_PROCESSING = "file_processing"
    AGENT_COMMUNICATION = "agent_communication"
    SECURITY_VALIDATION = "security_validation"
    NETWORK_CONNECTIVITY = "network_connectivity"
    DATA_VALIDATION = "data_validation"
    UI_INTERACTION = "ui_interaction"
    SYSTEM_RESOURCE = "system_resource"
    AUTHENTICATION = "authentication"
    UNKNOWN = "unknown"


@dataclass
class ErrorContext:
    """오류 컨텍스트 정보"""
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    operation: Optional[str] = None
    file_name: Optional[str] = None
    agent_name: Optional[str] = None
    request_data: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.now)


# Custom Exception Classes
class CherryAIException(Exception):
    """Base exception for Cherry AI application"""
    def __init__(self, message: str, category: ErrorCategory = ErrorCategory.UNKNOWN, 
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM, context: Optional[ErrorContext] = None):
        super().__init__(message)
        self.category = category
        self.severity = severity
        self.context = context or ErrorContext()
        self.timestamp = datetime.now()


class FileProcessingError(CherryAIException):
    """File processing related errors"""
    def __init__(self, message: str, file_name: str = None, **kwargs):
        context = ErrorContext(file_name=file_name, operation="file_processing")
        super().__init__(message, ErrorCategory.FILE_PROCESSING, **kwargs, context=context)


class AgentCommunicationError(CherryAIException):
    """Agent communication errors"""
    def __init__(self, message: str, agent_name: str = None, **kwargs):
        context = ErrorContext(agent_name=agent_name, operation="agent_communication")
        super().__init__(message, ErrorCategory.AGENT_COMMUNICATION, **kwargs, context=context)


class SecurityValidationError(CherryAIException):
    """Security validation errors"""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, ErrorCategory.SECURITY_VALIDATION, ErrorSeverity.HIGH, **kwargs)


class NetworkConnectivityError(CherryAIException):
    """Network connectivity errors"""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, ErrorCategory.NETWORK_CONNECTIVITY, ErrorSeverity.MEDIUM, **kwargs)


class DataValidationError(CherryAIException):
    """Data validation errors"""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, ErrorCategory.DATA_VALIDATION, ErrorSeverity.MEDIUM, **kwargs)


class UIInteractionError(CherryAIException):
    """UI interaction errors"""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, ErrorCategory.UI_INTERACTION, ErrorSeverity.LOW, **kwargs)


class SystemResourceError(CherryAIException):
    """System resource errors"""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, ErrorCategory.SYSTEM_RESOURCE, ErrorSeverity.HIGH, **kwargs)


class AuthenticationError(CherryAIException):
    """Authentication/authorization errors"""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, ErrorCategory.AUTHENTICATION, ErrorSeverity.HIGH, **kwargs)


class EnhancedExceptionHandler:
    """
    향상된 예외 처리 시스템
    - 특정 예외 타입별 처리
    - 사용자 친화적 메시지
    - 자동 복구 시도
    - 상세한 로깅
    """
    
    def __init__(self):
        self.error_handlers: Dict[Type[Exception], Callable] = {}
        self.recovery_strategies: Dict[ErrorCategory, Callable] = {}
        self.user_message_templates: Dict[ErrorCategory, str] = {}
        self.error_statistics: Dict[str, int] = {}
        
        self._setup_default_handlers()
        self._setup_user_messages()
        
        logger.info("Enhanced Exception Handler initialized")
    
    def _setup_default_handlers(self):
        """기본 오류 핸들러 설정"""
        self.error_handlers = {
            FileNotFoundError: self._handle_file_not_found,
            PermissionError: self._handle_permission_error,
            ConnectionError: self._handle_connection_error,
            TimeoutError: self._handle_timeout_error,
            ValueError: self._handle_value_error,
            KeyError: self._handle_key_error,
            ImportError: self._handle_import_error,
            asyncio.TimeoutError: self._handle_async_timeout,
            
            # Custom exceptions
            FileProcessingError: self._handle_file_processing_error,
            AgentCommunicationError: self._handle_agent_communication_error,
            SecurityValidationError: self._handle_security_validation_error,
            NetworkConnectivityError: self._handle_network_connectivity_error,
            DataValidationError: self._handle_data_validation_error,
            UIInteractionError: self._handle_ui_interaction_error,
            SystemResourceError: self._handle_system_resource_error,
            AuthenticationError: self._handle_authentication_error,
        }
    
    def _setup_user_messages(self):
        """사용자 친화적 메시지 템플릿"""
        self.user_message_templates = {
            ErrorCategory.FILE_PROCESSING: "📁 파일 처리 중 오류가 발생했습니다. 파일 형식과 크기를 확인해주세요.",
            ErrorCategory.AGENT_COMMUNICATION: "🤖 에이전트 통신 오류입니다. 잠시 후 다시 시도해주세요.",
            ErrorCategory.SECURITY_VALIDATION: "🛡️ 보안 검증에서 문제가 발견되었습니다. 입력 내용을 확인해주세요.",
            ErrorCategory.NETWORK_CONNECTIVITY: "🌐 네트워크 연결에 문제가 있습니다. 인터넷 연결을 확인해주세요.",
            ErrorCategory.DATA_VALIDATION: "📊 데이터 형식에 문제가 있습니다. 데이터를 확인해주세요.",
            ErrorCategory.UI_INTERACTION: "🖥️ 화면 상호작용 오류입니다. 페이지를 새로고침해주세요.",
            ErrorCategory.SYSTEM_RESOURCE: "⚡ 시스템 리소스 부족입니다. 잠시 후 다시 시도해주세요.",
            ErrorCategory.AUTHENTICATION: "🔐 인증 오류입니다. 권한을 확인해주세요.",
            ErrorCategory.UNKNOWN: "❓ 예상치 못한 오류가 발생했습니다. 관리자에게 문의해주세요."
        }
    
    async def handle_exception(self, exception: Exception, context: Optional[ErrorContext] = None) -> Dict[str, Any]:
        """
        예외 처리 메인 메서드
        특정 예외 타입에 따라 적절한 처리 수행
        """
        try:
            # 오류 통계 업데이트
            exception_type = type(exception).__name__
            self.error_statistics[exception_type] = self.error_statistics.get(exception_type, 0) + 1
            
            # 컨텍스트 정보 보강
            if context is None:
                context = ErrorContext()
            
            # 특정 핸들러 찾기
            handler = self._find_exception_handler(exception)
            
            if handler:
                result = await self._execute_handler(handler, exception, context)
            else:
                result = await self._handle_generic_exception(exception, context)
            
            # 로깅
            await self._log_exception(exception, context, result)
            
            return result
            
        except Exception as handler_error:
            logger.error(f"Error in exception handler: {str(handler_error)}")
            return await self._handle_handler_failure(exception, handler_error, context)
    
    def _find_exception_handler(self, exception: Exception) -> Optional[Callable]:
        """예외 타입에 맞는 핸들러 찾기"""
        for exception_type, handler in self.error_handlers.items():
            if isinstance(exception, exception_type):
                return handler
        return None
    
    async def _execute_handler(self, handler: Callable, exception: Exception, context: ErrorContext) -> Dict[str, Any]:
        """핸들러 실행"""
        if asyncio.iscoroutinefunction(handler):
            return await handler(exception, context)
        else:
            return handler(exception, context)
    
    # Specific Exception Handlers
    
    async def _handle_file_not_found(self, exception: FileNotFoundError, context: ErrorContext) -> Dict[str, Any]:
        """파일 없음 오류 처리"""
        return {
            "handled": True,
            "severity": ErrorSeverity.MEDIUM,
            "user_message": f"📁 파일을 찾을 수 없습니다: {context.file_name or '알 수 없는 파일'}",
            "developer_message": str(exception),
            "recovery_action": "파일 경로와 권한을 확인해주세요.",
            "retry_possible": True
        }
    
    async def _handle_permission_error(self, exception: PermissionError, context: ErrorContext) -> Dict[str, Any]:
        """권한 오류 처리"""
        return {
            "handled": True,
            "severity": ErrorSeverity.HIGH,
            "user_message": "🔐 파일 접근 권한이 없습니다. 관리자에게 문의해주세요.",
            "developer_message": str(exception),
            "recovery_action": "파일 권한을 확인하거나 다른 파일을 선택해주세요.",
            "retry_possible": False
        }
    
    async def _handle_connection_error(self, exception: ConnectionError, context: ErrorContext) -> Dict[str, Any]:
        """연결 오류 처리"""
        return {
            "handled": True,
            "severity": ErrorSeverity.MEDIUM,
            "user_message": "🌐 서버 연결에 실패했습니다. 잠시 후 다시 시도해주세요.",
            "developer_message": str(exception),
            "recovery_action": "네트워크 연결을 확인하고 잠시 후 재시도",
            "retry_possible": True,
            "retry_delay": 5
        }
    
    async def _handle_timeout_error(self, exception: TimeoutError, context: ErrorContext) -> Dict[str, Any]:
        """타임아웃 오류 처리"""
        return {
            "handled": True,
            "severity": ErrorSeverity.MEDIUM,
            "user_message": "⏱️ 작업 시간이 초과되었습니다. 다시 시도해주세요.",
            "developer_message": str(exception),
            "recovery_action": "더 작은 파일을 사용하거나 잠시 후 재시도",
            "retry_possible": True,
            "retry_delay": 3
        }
    
    async def _handle_value_error(self, exception: ValueError, context: ErrorContext) -> Dict[str, Any]:
        """값 오류 처리"""
        return {
            "handled": True,
            "severity": ErrorSeverity.MEDIUM,
            "user_message": "📊 입력된 데이터에 문제가 있습니다. 데이터 형식을 확인해주세요.",
            "developer_message": str(exception),
            "recovery_action": "올바른 형식의 데이터를 입력해주세요.",
            "retry_possible": True
        }
    
    async def _handle_key_error(self, exception: KeyError, context: ErrorContext) -> Dict[str, Any]:
        """키 오류 처리"""
        missing_key = str(exception).strip("'\"")
        return {
            "handled": True,
            "severity": ErrorSeverity.MEDIUM,
            "user_message": f"📋 필요한 데이터 필드가 없습니다: {missing_key}",
            "developer_message": str(exception),
            "recovery_action": "데이터에 필요한 컬럼이 포함되어 있는지 확인해주세요.",
            "retry_possible": True
        }
    
    async def _handle_import_error(self, exception: ImportError, context: ErrorContext) -> Dict[str, Any]:
        """Import 오류 처리"""
        return {
            "handled": True,
            "severity": ErrorSeverity.HIGH,
            "user_message": "⚙️ 시스템 구성요소를 불러올 수 없습니다. 관리자에게 문의해주세요.",
            "developer_message": str(exception),
            "recovery_action": "필요한 라이브러리가 설치되어 있는지 확인",
            "retry_possible": False
        }
    
    async def _handle_async_timeout(self, exception: asyncio.TimeoutError, context: ErrorContext) -> Dict[str, Any]:
        """비동기 타임아웃 처리"""
        return {
            "handled": True,
            "severity": ErrorSeverity.MEDIUM,
            "user_message": "⏱️ 서버 응답 시간이 초과되었습니다. 잠시 후 다시 시도해주세요.",
            "developer_message": str(exception),
            "recovery_action": "네트워크 상태를 확인하고 재시도",
            "retry_possible": True,
            "retry_delay": 5
        }
    
    # Custom Exception Handlers
    
    async def _handle_file_processing_error(self, exception: FileProcessingError, context: ErrorContext) -> Dict[str, Any]:
        """파일 처리 오류"""
        return {
            "handled": True,
            "severity": exception.severity,
            "user_message": f"📁 파일 처리 오류: {str(exception)}",
            "developer_message": str(exception),
            "recovery_action": "파일 형식과 내용을 확인해주세요.",
            "retry_possible": True
        }
    
    async def _handle_agent_communication_error(self, exception: AgentCommunicationError, context: ErrorContext) -> Dict[str, Any]:
        """에이전트 통신 오류"""
        return {
            "handled": True,
            "severity": exception.severity,
            "user_message": f"🤖 {context.agent_name or 'Agent'} 통신 오류: {str(exception)}",
            "developer_message": str(exception),
            "recovery_action": "에이전트 상태를 확인하고 재시도",
            "retry_possible": True,
            "retry_delay": 3
        }
    
    async def _handle_security_validation_error(self, exception: SecurityValidationError, context: ErrorContext) -> Dict[str, Any]:
        """보안 검증 오류"""
        return {
            "handled": True,
            "severity": ErrorSeverity.HIGH,
            "user_message": "🛡️ 보안 검증 실패: 안전하지 않은 내용이 감지되었습니다.",
            "developer_message": str(exception),
            "recovery_action": "입력 내용을 검토하고 안전한 데이터를 사용해주세요.",
            "retry_possible": False
        }
    
    async def _handle_network_connectivity_error(self, exception: NetworkConnectivityError, context: ErrorContext) -> Dict[str, Any]:
        """네트워크 연결 오류"""
        return {
            "handled": True,
            "severity": exception.severity,
            "user_message": "🌐 네트워크 연결 오류가 발생했습니다.",
            "developer_message": str(exception),
            "recovery_action": "인터넷 연결을 확인하고 재시도",
            "retry_possible": True,
            "retry_delay": 5
        }
    
    async def _handle_data_validation_error(self, exception: DataValidationError, context: ErrorContext) -> Dict[str, Any]:
        """데이터 검증 오류"""
        return {
            "handled": True,
            "severity": exception.severity,
            "user_message": f"📊 데이터 검증 오류: {str(exception)}",
            "developer_message": str(exception),
            "recovery_action": "데이터 형식과 내용을 확인해주세요.",
            "retry_possible": True
        }
    
    async def _handle_ui_interaction_error(self, exception: UIInteractionError, context: ErrorContext) -> Dict[str, Any]:
        """UI 상호작용 오류"""
        return {
            "handled": True,
            "severity": ErrorSeverity.LOW,
            "user_message": "🖥️ 화면 오류가 발생했습니다. 페이지를 새로고침해주세요.",
            "developer_message": str(exception),
            "recovery_action": "페이지 새로고침 또는 브라우저 재시작",
            "retry_possible": True
        }
    
    async def _handle_system_resource_error(self, exception: SystemResourceError, context: ErrorContext) -> Dict[str, Any]:
        """시스템 리소스 오류"""
        return {
            "handled": True,
            "severity": ErrorSeverity.HIGH,
            "user_message": "⚡ 시스템 리소스가 부족합니다. 잠시 후 다시 시도해주세요.",
            "developer_message": str(exception),
            "recovery_action": "더 작은 데이터를 사용하거나 나중에 재시도",
            "retry_possible": True,
            "retry_delay": 10
        }
    
    async def _handle_authentication_error(self, exception: AuthenticationError, context: ErrorContext) -> Dict[str, Any]:
        """인증 오류"""
        return {
            "handled": True,
            "severity": ErrorSeverity.HIGH,
            "user_message": "🔐 인증에 실패했습니다. 권한을 확인해주세요.",
            "developer_message": str(exception),
            "recovery_action": "로그인 상태와 권한을 확인",
            "retry_possible": False
        }
    
    async def _handle_generic_exception(self, exception: Exception, context: ErrorContext) -> Dict[str, Any]:
        """일반 예외 처리"""
        return {
            "handled": False,
            "severity": ErrorSeverity.MEDIUM,
            "user_message": f"❓ 예상치 못한 오류가 발생했습니다: {type(exception).__name__}",
            "developer_message": str(exception),
            "recovery_action": "페이지를 새로고침하거나 관리자에게 문의해주세요.",
            "retry_possible": True
        }
    
    async def _handle_handler_failure(self, original_exception: Exception, handler_error: Exception, context: ErrorContext) -> Dict[str, Any]:
        """핸들러 실행 실패 처리"""
        logger.critical(f"Exception handler failed: {str(handler_error)}")
        return {
            "handled": False,
            "severity": ErrorSeverity.CRITICAL,
            "user_message": "❌ 시스템 오류가 발생했습니다. 관리자에게 즉시 문의해주세요.",
            "developer_message": f"Original: {str(original_exception)}, Handler Error: {str(handler_error)}",
            "recovery_action": "시스템 재시작이 필요할 수 있습니다.",
            "retry_possible": False
        }
    
    async def _log_exception(self, exception: Exception, context: ErrorContext, result: Dict[str, Any]):
        """예외 로깅"""
        log_data = {
            "exception_type": type(exception).__name__,
            "message": str(exception),
            "context": {
                "user_id": context.user_id,
                "session_id": context.session_id,
                "operation": context.operation,
                "file_name": context.file_name,
                "agent_name": context.agent_name,
                "timestamp": context.timestamp.isoformat()
            },
            "handling_result": {
                "handled": result.get("handled", False),
                "severity": result.get("severity", ErrorSeverity.UNKNOWN).value if hasattr(result.get("severity"), "value") else str(result.get("severity")),
                "retry_possible": result.get("retry_possible", False)
            },
            "traceback": traceback.format_exc()
        }
        
        severity = result.get("severity", ErrorSeverity.MEDIUM)
        if severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            logger.error(f"Exception handled: {log_data}")
        else:
            logger.warning(f"Exception handled: {log_data}")
    
    def display_user_error(self, result: Dict[str, Any]):
        """Streamlit에 사용자 친화적 오류 메시지 표시"""
        severity = result.get("severity", ErrorSeverity.MEDIUM)
        user_message = result.get("user_message", "알 수 없는 오류가 발생했습니다.")
        recovery_action = result.get("recovery_action", "")
        
        if severity == ErrorSeverity.CRITICAL:
            st.error(f"{user_message}\n\n**복구 방법:** {recovery_action}")
        elif severity == ErrorSeverity.HIGH:
            st.error(f"{user_message}\n\n**해결 방법:** {recovery_action}")
        elif severity == ErrorSeverity.MEDIUM:
            st.warning(f"{user_message}\n\n**권장 사항:** {recovery_action}")
        else:
            st.info(f"{user_message}")
        
        # 재시도 가능한 경우 버튼 표시
        if result.get("retry_possible", False):
            if st.button("🔄 다시 시도"):
                st.rerun()
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """오류 통계 조회"""
        return {
            "total_errors": sum(self.error_statistics.values()),
            "error_breakdown": self.error_statistics.copy(),
            "most_common_error": max(self.error_statistics.items(), key=lambda x: x[1])[0] if self.error_statistics else None
        }


# 전역 예외 핸들러 인스턴스
_global_exception_handler: Optional[EnhancedExceptionHandler] = None


def get_exception_handler() -> EnhancedExceptionHandler:
    """전역 예외 핸들러 반환"""
    global _global_exception_handler
    if _global_exception_handler is None:
        _global_exception_handler = EnhancedExceptionHandler()
    return _global_exception_handler


# Decorator for automatic exception handling
def handle_exceptions(operation_name: str = None):
    """자동 예외 처리 데코레이터"""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                handler = get_exception_handler()
                context = ErrorContext(operation=operation_name or func.__name__)
                result = await handler.handle_exception(e, context)
                handler.display_user_error(result)
                return None
        
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                handler = get_exception_handler()
                context = ErrorContext(operation=operation_name or func.__name__)
                result = asyncio.run(handler.handle_exception(e, context))
                handler.display_user_error(result)
                return None
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator