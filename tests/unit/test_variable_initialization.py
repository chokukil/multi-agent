"""
변수 초기화 오류 테스트 모듈

A2A 에이전트들의 변수 초기화 문제를 체계적으로 테스트
"""

import pytest
import os
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# 테스트 데이터 경로
TEST_DATA_PATH = "tests/fixtures"

class TestVariableInitialization:
    """변수 초기화 오류 테스트 클래스"""
    
    def setup_method(self):
        """테스트 전 설정"""
        self.test_df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c']
        })
        
        # 테스트 데이터 폴더 생성
        os.makedirs(TEST_DATA_PATH, exist_ok=True)
        test_csv_path = os.path.join(TEST_DATA_PATH, "test.csv")
        self.test_df.to_csv(test_csv_path, index=False)
    
    def test_data_file_initialization_error(self):
        """data_file 변수 초기화 오류 재현 테스트"""
        # H2O ML 서버의 패턴을 재현
        available_data = ["test.csv", "test2.csv"]
        
        # 현재 문제점: data_file이 초기화되지 않음
        data_file = None  # 실제 서버에서 발생하는 상황
        
        with pytest.raises(AttributeError):
            # 이 코드는 에러를 발생시킴
            if data_file.endswith('.csv'):
                df = pd.read_csv(data_file)
    
    def test_df_initialization_error(self):
        """df 변수 초기화 오류 재현 테스트"""
        # Data Cleaning 서버의 패턴을 재현
        df = None  # 초기화되지 않은 상태
        
        with pytest.raises(AttributeError):
            # 이 코드는 에러를 발생시킴
            shape = df.shape
    
    def test_safe_data_file_initialization(self):
        """안전한 data_file 초기화 패턴 테스트"""
        available_data = ["test.csv", "test2.csv"]
        
        # 안전한 초기화 패턴
        data_file = None
        
        if available_data:
            data_file = available_data[0]
        
        # 안전한 사용 패턴
        if data_file and data_file.endswith('.csv'):
            # 이 코드는 정상 작동
            assert data_file == "test.csv"
    
    def test_safe_df_initialization(self):
        """안전한 df 초기화 패턴 테스트"""
        # 안전한 초기화 패턴
        df = None
        
        try:
            df = pd.DataFrame({'test': [1, 2, 3]})
        except Exception:
            pass
        
        # 안전한 사용 패턴
        if df is not None:
            assert df.shape == (3, 1)
    
    def test_unified_data_loading_pattern(self):
        """통합된 데이터 로딩 패턴 테스트"""
        # 통합된 안전한 패턴
        def safe_load_data(data_path, available_data):
            """안전한 데이터 로딩 함수"""
            df = None
            data_file = None
            error_msg = None
            
            try:
                if not available_data:
                    error_msg = "No data available"
                    return df, data_file, error_msg
                
                # 안전한 파일 선택
                data_file = available_data[0]
                
                if not data_file:
                    error_msg = "No valid data file selected"
                    return df, data_file, error_msg
                
                # 안전한 데이터 로딩
                file_path = os.path.join(data_path, data_file)
                
                if not os.path.exists(file_path):
                    error_msg = f"File not found: {file_path}"
                    return df, data_file, error_msg
                
                if data_file.endswith('.csv'):
                    df = pd.read_csv(file_path)
                elif data_file.endswith('.pkl'):
                    df = pd.read_pickle(file_path)
                else:
                    error_msg = f"Unsupported file format: {data_file}"
                    return df, data_file, error_msg
                
                return df, data_file, None
                
            except Exception as e:
                error_msg = f"Error loading data: {str(e)}"
                return df, data_file, error_msg
        
        # 테스트 실행
        df, data_file, error_msg = safe_load_data(TEST_DATA_PATH, ["test.csv"])
        
        assert df is not None
        assert data_file == "test.csv"
        assert error_msg is None
        assert df.shape == (3, 2)
    
    def test_empty_data_handling(self):
        """빈 데이터 처리 테스트"""
        def safe_load_data(data_path, available_data):
            df = None
            data_file = None
            error_msg = None
            
            if not available_data:
                error_msg = "No data files available"
                return df, data_file, error_msg
            
            return df, data_file, error_msg
        
        # 빈 데이터 테스트
        df, data_file, error_msg = safe_load_data(TEST_DATA_PATH, [])
        
        assert df is None
        assert data_file is None
        assert "No data files available" in error_msg
    
    def test_file_not_found_handling(self):
        """파일 없음 처리 테스트"""
        def safe_load_data(data_path, available_data):
            df = None
            data_file = None
            error_msg = None
            
            if not available_data:
                error_msg = "No data available"
                return df, data_file, error_msg
            
            data_file = available_data[0]
            file_path = os.path.join(data_path, data_file)
            
            if not os.path.exists(file_path):
                error_msg = f"File not found: {file_path}"
                return df, data_file, error_msg
            
            return df, data_file, error_msg
        
        # 존재하지 않는 파일 테스트
        df, data_file, error_msg = safe_load_data(TEST_DATA_PATH, ["nonexistent.csv"])
        
        assert df is None
        assert data_file == "nonexistent.csv"
        assert "File not found" in error_msg

class TestDataAccessPatterns:
    """데이터 접근 패턴 테스트"""
    
    def test_multiple_data_sources(self):
        """다중 데이터 소스 처리 테스트"""
        # 다중 데이터 소스 시뮬레이션
        data_sources = [
            {"name": "sales.csv", "priority": 1},
            {"name": "customers.csv", "priority": 2},
            {"name": "products.csv", "priority": 3}
        ]
        
        # 우선순위 기반 선택
        selected_source = min(data_sources, key=lambda x: x["priority"])
        
        assert selected_source["name"] == "sales.csv"
        assert selected_source["priority"] == 1
    
    def test_data_context_preservation(self):
        """데이터 컨텍스트 보존 테스트"""
        # 컨텍스트 정보
        context = {
            "session_id": "test_session",
            "user_request": "분석해줘",
            "selected_file": "test.csv",
            "timestamp": "2024-01-01T00:00:00"
        }
        
        # 컨텍스트 보존 확인
        assert context["session_id"] == "test_session"
        assert context["selected_file"] == "test.csv"
        assert "timestamp" in context
    
    def teardown_method(self):
        """테스트 후 정리"""
        # 테스트 파일 삭제
        test_csv_path = os.path.join(TEST_DATA_PATH, "test.csv")
        if os.path.exists(test_csv_path):
            os.remove(test_csv_path) 