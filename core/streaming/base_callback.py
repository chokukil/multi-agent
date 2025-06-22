# core/streaming/base_callback.py
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Callable
import logging

class BaseStreamCallback(ABC):
    """스트림 콜백의 기본 추상 클래스"""
    
    def __init__(self, 
                 ui_container,
                 progress_callback: Optional[Callable] = None,
                 error_callback: Optional[Callable] = None):
        self.ui_container = ui_container
        self.progress_callback = progress_callback
        self.error_callback = error_callback
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def __call__(self, msg_data: Any):
        """메시지 처리 진입점"""
        pass
    
    def log_debug(self, message: str):
        """디버그 로깅"""
        print(f"🔍 [DEBUG] {self.__class__.__name__}: {message}")
        self.logger.debug(message)
    
    def log_info(self, message: str):
        """정보 로깅"""
        print(f"ℹ️ [INFO] {self.__class__.__name__}: {message}")
        self.logger.info(message)
    
    def log_warning(self, message: str):
        """경고 로깅"""
        print(f"⚠️ [WARNING] {self.__class__.__name__}: {message}")
        self.logger.warning(message)
    
    def log_error(self, message: str):
        """에러 로깅"""
        print(f"❌ [ERROR] {self.__class__.__name__}: {message}")
        self.logger.error(message)