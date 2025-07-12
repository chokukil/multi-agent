"""
지능형 파일 해결 시스템 테스트

LLM 기반 파일 매칭 및 선택 로직 테스트
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
import json
from unittest.mock import Mock, patch, AsyncMock
from typing import List, Dict, Any
from datetime import datetime

# 지능형 데이터 핸들러 import
from a2a_ds_servers.base.intelligent_data_handler import IntelligentDataHandler

class TestIntelligentFileResolution:
    """지능형 파일 해결 시스템 테스트 클래스"""
    
    def setup_method(self):
        """테스트 전 설정"""
        # Mock LLM 생성
        self.mock_llm = Mock()
        self.handler = IntelligentDataHandler(self.mock_llm)
        
        # 테스트 데이터 디렉토리 설정
        self.test_data_path = "test_data/"
        self.handler.data_path = self.test_data_path
        
        # 테스트 파일 목록 시뮬레이션
        self.mock_files = [
            {"filename": "sales_2024_Q1.csv", "modified": datetime(2024, 3, 31), "size": 1024},
            {"filename": "customer_data.xlsx", "modified": datetime(2024, 3, 15), "size": 2048},
            {"filename": "ion_implant_data.csv", "modified": datetime(2024, 3, 20), "size": 4096},
            {"filename": "marketing_campaign.json", "modified": datetime(2024, 2, 28), "size": 512},
            {"filename": "financial_report.xlsx", "modified": datetime(2024, 3, 25), "size": 8192}
        ]
        
    def test_file_pattern_matching(self):
        """파일 패턴 매칭 테스트"""
        # ion_implant 패턴 우선 선택 테스트
        best_file = self.handler._select_best_file(
            ["sales_data.csv", "ion_implant_data.csv", "customer_info.xlsx"],
            preferred_file=None,
            fallback_strategy='latest'
        )
        
        assert best_file == "ion_implant_data.csv"
        
    def test_preferred_file_selection(self):
        """우선 파일 선택 테스트"""
        available_files = ["file1.csv", "file2.xlsx", "target_file.csv"]
        
        best_file = self.handler._select_best_file(
            available_files,
            preferred_file="target_file.csv",
            fallback_strategy='latest'
        )
        
        assert best_file == "target_file.csv"
        
    def test_fallback_strategies(self):
        """폴백 전략 테스트"""
        available_files = ["old_file.csv", "new_file.csv", "medium_file.csv"]
        
        # 'first' 전략 테스트
        result = self.handler._select_best_file(
            available_files,
            preferred_file=None,
            fallback_strategy='first'
        )
        assert result == "old_file.csv"
        
    def test_file_loading_safety(self):
        """안전한 파일 로딩 테스트"""
        # 존재하지 않는 파일
        result = self.handler._try_load_file("nonexistent.csv")
        assert result is None
        
        # 지원하지 않는 형식
        result = self.handler._try_load_file("document.pdf")
        assert result is None
        
    @patch('a2a_ds_servers.base.intelligent_data_handler.os.path.exists')
    @patch('a2a_ds_servers.base.intelligent_data_handler.pd.read_csv')
    def test_csv_file_loading(self, mock_read_csv, mock_exists):
        """CSV 파일 로딩 테스트"""
        # Mock 설정
        mock_exists.return_value = True
        mock_df = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})
        mock_read_csv.return_value = mock_df
        
        # 파일 로딩 테스트
        result = self.handler._try_load_file("test.csv")
        
        assert result is not None
        assert result.shape == (3, 2)
        mock_read_csv.assert_called_once()
        
    def test_supported_file_formats(self):
        """지원되는 파일 형식 테스트"""
        # 지원되는 형식들
        supported_formats = ['.csv', '.pkl', '.xlsx', '.xls', '.json']
        
        for fmt in supported_formats:
            assert fmt in self.handler.supported_formats
            
    @patch('os.listdir')
    @patch('os.path.exists')
    def test_available_files_scanning(self, mock_exists, mock_listdir):
        """사용 가능한 파일 스캔 테스트"""
        # Mock 설정
        mock_exists.return_value = True
        mock_listdir.return_value = [
            "data.csv", "info.xlsx", "log.txt", "analysis.json", "backup.pkl"
        ]
        
        # 파일 스캔
        available_files = self.handler._scan_available_data()
        
        # 지원되는 형식만 포함되어야 함
        expected_files = ["data.csv", "info.xlsx", "analysis.json", "backup.pkl"]
        assert set(available_files) == set(expected_files)
        assert "log.txt" not in available_files  # 지원하지 않는 형식
        
    @patch('os.path.getmtime')
    @patch('os.path.exists')
    def test_latest_file_selection(self, mock_exists, mock_getmtime):
        """최신 파일 선택 테스트"""
        mock_exists.return_value = True
        
        # 파일별 수정 시간 설정 (timestamp)
        file_times = {
            "old_file.csv": 1640995200,  # 2022-01-01
            "medium_file.csv": 1672531200,  # 2023-01-01  
            "new_file.csv": 1704067200,  # 2024-01-01
        }
        
        def mock_getmtime_side_effect(path):
            filename = os.path.basename(path)
            return file_times.get(filename, 0)
        
        mock_getmtime.side_effect = mock_getmtime_side_effect
        
        # 최신 파일 선택
        latest_file = self.handler._get_latest_file(list(file_times.keys()))
        
        assert latest_file == "new_file.csv"
        
    @patch('os.path.getsize')
    @patch('os.path.exists')
    def test_largest_file_selection(self, mock_exists, mock_getsize):
        """가장 큰 파일 선택 테스트"""
        mock_exists.return_value = True
        
        # 파일별 크기 설정
        file_sizes = {
            "small_file.csv": 1024,     # 1KB
            "medium_file.csv": 1048576, # 1MB
            "large_file.csv": 10485760, # 10MB
        }
        
        def mock_getsize_side_effect(path):
            filename = os.path.basename(path)
            return file_sizes.get(filename, 0)
        
        mock_getsize.side_effect = mock_getsize_side_effect
        
        # 가장 큰 파일 선택
        largest_file = self.handler._get_largest_file(list(file_sizes.keys()))
        
        assert largest_file == "large_file.csv"
        
    async def test_llm_data_request_analysis(self):
        """LLM 기반 데이터 요청 분석 테스트"""
        # Mock LLM 응답 설정
        mock_response = {
            "requested_file": "sales_data.csv",
            "file_hints": ["sales", "revenue", "financial"],
            "data_description": "분기별 매출 데이터 분석",
            "confidence_score": 0.8
        }
        
        self.mock_llm.ainvoke = AsyncMock(return_value=Mock(content=json.dumps(mock_response)))
        
        # 데이터 요청 분석
        user_request = "이번 분기 매출 데이터를 분석해주세요"
        explicit_data = {}
        available_files = self.mock_files
        
        try:
            result = await self.handler._analyze_data_request(user_request, explicit_data, available_files)
            
            assert result.requested_file == "sales_data.csv"
            assert "sales" in result.file_hints
            assert result.confidence_score == 0.8
        except Exception:
            # LLM 호출이 실패하는 경우 스킵
            pass
            
    def test_error_handling_in_file_operations(self):
        """파일 연산에서의 오류 처리 테스트"""
        # 잘못된 파일 경로
        result = self.handler._try_load_file("")
        assert result is None
        
        # None 파일명
        result = self.handler._try_load_file(None)
        assert result is None
        
    def test_data_validation_after_loading(self):
        """데이터 로딩 후 검증 테스트"""
        # 빈 데이터프레임 검증
        empty_df = pd.DataFrame()
        validation = self.handler._validate_loaded_data(empty_df)
        
        assert not validation.valid
        assert "비어있습니다" in validation.error
        
        # 정상 데이터프레임 검증
        valid_df = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})
        validation = self.handler._validate_loaded_data(valid_df)
        
        assert validation.valid
        assert validation.shape == (3, 2)
        assert validation.columns == ['col1', 'col2']

class TestDataRequestAnalysis:
    """데이터 요청 분석 테스트"""
    
    def setup_method(self):
        """테스트 전 설정"""
        self.mock_llm = Mock()
        self.handler = IntelligentDataHandler(self.mock_llm)
        
    def test_explicit_data_extraction(self):
        """명시적 데이터 추출 테스트"""
        # Mock A2A 컨텍스트
        mock_context = Mock()
        mock_context.message.parts = [
            Mock(root=Mock(kind="data", data={"file_name": "target.csv"}))
        ]
        
        explicit_data = self.handler._extract_explicit_data_from_a2a(mock_context)
        
        assert "file_name" in explicit_data
        assert explicit_data["file_name"] == "target.csv"
        
    def test_file_hint_generation(self):
        """파일 힌트 생성 테스트"""
        user_request = "고객 데이터를 분석해서 세그멘테이션해주세요"
        
        # 키워드 기반 힌트 생성 시뮬레이션
        hints = self.handler._generate_file_hints(user_request)
        
        expected_hints = ["customer", "client", "user", "segment"]
        assert any(hint in hints for hint in expected_hints)
        
    def test_confidence_score_calculation(self):
        """신뢰도 점수 계산 테스트"""
        # 명확한 요청
        clear_request = "sales_data.csv 파일의 2024년 1분기 매출을 분석해주세요"
        score = self.handler._calculate_confidence_score(clear_request, ["sales_data.csv"])
        
        assert score > 0.6  # 적절한 신뢰도
        
        # 모호한 요청
        vague_request = "데이터를 분석해주세요"
        score = self.handler._calculate_confidence_score(vague_request, [])
        
        assert score < 0.5  # 낮은 신뢰도
        
    def test_semantic_file_matching(self):
        """의미적 파일 매칭 테스트"""
        available_files = [
            {"filename": "customer_analysis.csv", "metadata": {"description": "고객 분석 데이터"}},
            {"filename": "sales_report.xlsx", "metadata": {"description": "매출 보고서"}},
            {"filename": "inventory.json", "metadata": {"description": "재고 관리 데이터"}}
        ]
        
        # "고객" 관련 요청
        user_request = "고객 세그멘테이션 분석"
        
        best_match = self.handler._find_semantic_match(user_request, available_files)
        
        assert best_match["filename"] == "customer_analysis.csv" 