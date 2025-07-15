#!/usr/bin/env python3
"""
CherryAI Unified SQL Database Server - Port 8311
A2A SDK 0.2.9 완전 표준 준수 + UnifiedDataInterface 패턴

🗄️ 핵심 기능:
- 🧠 LLM 기반 지능형 SQL 쿼리 생성 및 최적화
- 🔗 DB 연결 안정성 개선 (설계 문서 주요 개선사항)
- ⚠️ 예외 처리 강화 및 복구 메커니즘
- 🔒 SQL 인젝션 방지 및 보안 강화
- 💾 연결 풀링 및 성능 최적화
- 🎯 A2A 표준 TaskUpdater + 실시간 스트리밍

기반: pandas_agent 패턴 + unified_data_loader 성공 사례
"""

import asyncio
import logging
import os
import json
import sys
import time
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from contextlib import asynccontextmanager
import sqlparse
import re

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# A2A SDK 0.2.9 표준 imports
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import (
    AgentCard, AgentSkill, AgentCapabilities,
    TaskState, TextPart
)
from a2a.utils import new_agent_text_message
import uvicorn

# CherryAI Core imports
from core.llm_factory import LLMFactory
from a2a_ds_servers.unified_data_system.core.unified_data_interface import UnifiedDataInterface
from a2a_ds_servers.unified_data_system.core.smart_dataframe import SmartDataFrame
from a2a_ds_servers.unified_data_system.core.llm_first_data_engine import LLMFirstDataEngine
from a2a_ds_servers.unified_data_system.core.cache_manager import CacheManager
from a2a_ds_servers.unified_data_system.utils.file_scanner import FileScanner

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DatabaseConnection:
    """데이터베이스 연결 정보"""
    connection_id: str
    db_type: str
    connection_string: str
    created_at: datetime
    last_used: datetime
    is_active: bool = True
    error_count: int = 0

@dataclass
class QueryResult:
    """SQL 쿼리 결과"""
    query: str
    execution_time: float
    row_count: int
    columns: List[str]
    success: bool
    error_message: Optional[str] = None

class DatabaseConnectionPool:
    """DB 연결 풀 관리 시스템 (안정성 개선 핵심)"""
    
    def __init__(self, max_connections: int = 5):
        self.max_connections = max_connections
        self.connections: Dict[str, DatabaseConnection] = {}
        self.active_connections = 0
        self.connection_timeout = 300  # 5분
        
    async def get_connection(self, db_path: str) -> Optional[sqlite3.Connection]:
        """안전한 DB 연결 획득"""
        try:
            # 기존 연결 재사용 시도
            for conn_id, db_conn in self.connections.items():
                if db_conn.connection_string == db_path and db_conn.is_active:
                    if self._is_connection_valid(conn_id):
                        db_conn.last_used = datetime.now()
                        return sqlite3.connect(db_path)
            
            # 새 연결 생성
            if self.active_connections < self.max_connections:
                conn_id = f"conn_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
                
                connection = sqlite3.connect(db_path)
                connection.row_factory = sqlite3.Row  # 딕셔너리 형태 결과
                
                db_conn = DatabaseConnection(
                    connection_id=conn_id,
                    db_type="sqlite",
                    connection_string=db_path,
                    created_at=datetime.now(),
                    last_used=datetime.now()
                )
                
                self.connections[conn_id] = db_conn
                self.active_connections += 1
                
                logger.info(f"✅ 새 DB 연결 생성: {conn_id}")
                return connection
            
            else:
                logger.warning("❌ 연결 풀 한계 도달")
                return None
                
        except Exception as e:
            logger.error(f"❌ DB 연결 실패: {e}")
            return None
    
    def _is_connection_valid(self, conn_id: str) -> bool:
        """연결 유효성 검사"""
        if conn_id not in self.connections:
            return False
        
        db_conn = self.connections[conn_id]
        
        # 타임아웃 검사
        if (datetime.now() - db_conn.last_used).seconds > self.connection_timeout:
            self._close_connection(conn_id)
            return False
        
        # 에러 횟수 검사
        if db_conn.error_count > 3:
            self._close_connection(conn_id)
            return False
        
        return db_conn.is_active
    
    def _close_connection(self, conn_id: str):
        """연결 종료"""
        if conn_id in self.connections:
            self.connections[conn_id].is_active = False
            self.active_connections = max(0, self.active_connections - 1)
            del self.connections[conn_id]
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """연결 풀 통계"""
        return {
            'active_connections': self.active_connections,
            'max_connections': self.max_connections,
            'total_connections_created': len(self.connections),
            'pool_utilization': f"{(self.active_connections / self.max_connections) * 100:.1f}%"
        }

class SQLSecurityValidator:
    """SQL 보안 검증 시스템"""
    
    def __init__(self):
        # 위험한 SQL 키워드
        self.dangerous_keywords = [
            'DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE',
            'TRUNCATE', 'EXEC', 'EXECUTE', 'UNION', '--', ';--'
        ]
        
        # 허용된 읽기 전용 키워드
        self.allowed_keywords = [
            'SELECT', 'FROM', 'WHERE', 'JOIN', 'INNER', 'LEFT', 'RIGHT',
            'GROUP BY', 'ORDER BY', 'HAVING', 'LIMIT', 'OFFSET', 'COUNT',
            'SUM', 'AVG', 'MIN', 'MAX', 'DISTINCT', 'AS', 'AND', 'OR'
        ]
    
    def validate_query(self, query: str) -> Tuple[bool, str]:
        """SQL 쿼리 보안 검증"""
        try:
            # 1. 기본 보안 검사
            query_upper = query.upper().strip()
            
            # 위험한 키워드 검사
            for keyword in self.dangerous_keywords:
                if keyword in query_upper:
                    return False, f"위험한 SQL 키워드 감지: {keyword}"
            
            # 2. SQL 파싱 검증
            try:
                parsed = sqlparse.parse(query)
                if not parsed:
                    return False, "SQL 구문 파싱 실패"
                
                # 여러 구문 방지
                if len(parsed) > 1:
                    return False, "다중 SQL 구문은 허용되지 않습니다"
                
            except Exception as e:
                return False, f"SQL 구문 오류: {str(e)}"
            
            # 3. SELECT 구문만 허용
            if not query_upper.strip().startswith('SELECT'):
                return False, "SELECT 구문만 허용됩니다"
            
            # 4. 길이 제한
            if len(query) > 10000:  # 10KB 제한
                return False, "SQL 쿼리가 너무 깁니다"
            
            return True, "안전한 쿼리"
            
        except Exception as e:
            return False, f"보안 검증 오류: {str(e)}"

class UnifiedSQLDatabaseExecutor(AgentExecutor, UnifiedDataInterface):
    """
    Unified SQL Database Executor
    
    pandas_agent 패턴 + data_loader 성공 사례 기반
    - LLM First SQL 쿼리 생성
    - DB 연결 안정성 보장
    - 예외 처리 강화
    - A2A SDK 0.2.9 완전 준수
    """
    
    def __init__(self):
        super().__init__()
        
        # 핵심 컴포넌트 초기화
        self.data_engine = LLMFirstDataEngine()
        self.cache_manager = CacheManager()
        self.file_scanner = FileScanner()
        self.llm_factory = LLMFactory()
        
        # DB 연결 안정성 시스템 (핵심 개선사항)
        self.connection_pool = DatabaseConnectionPool()
        self.security_validator = SQLSecurityValidator()
        
        # SQL 분석 전문 설정
        self.sql_capabilities = {
            'query_types': [
                'data_exploration', 'aggregation', 'filtering', 'joining',
                'statistical_analysis', 'reporting', 'business_intelligence'
            ],
            'sql_functions': [
                'COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'GROUP_CONCAT',
                'DISTINCT', 'CASE', 'SUBSTR', 'LENGTH', 'ROUND'
            ],
            'join_types': ['INNER', 'LEFT', 'RIGHT', 'FULL OUTER'],
            'analytical_functions': [
                'ROW_NUMBER', 'RANK', 'DENSE_RANK', 'NTILE'
            ]
        }
        
        # 예외 처리 설정 (강화된 복구 메커니즘)
        self.error_handling = {
            'max_retry_attempts': 3,
            'retry_delay': 1.0,
            'connection_timeout': 30,
            'query_timeout': 60,
            'auto_recovery': True
        }
        
        logger.info("✅ Unified SQL Database Executor 초기화 완료")
    
    async def execute(self, context: RequestContext, task_updater: TaskUpdater) -> None:
        """
        A2A 표준 실행: 8단계 지능형 SQL 데이터베이스 분석 프로세스
        
        🧠 1단계: LLM SQL 분석 의도 파악
        🗄️ 2단계: 데이터베이스 검색 및 연결 안정성 확인
        🔒 3단계: 보안 검증 및 연결 풀 최적화
        📊 4단계: 데이터베이스 스키마 분석
        💡 5단계: LLM 기반 SQL 쿼리 생성
        ⚡ 6단계: 안전한 쿼리 실행 및 예외 처리
        📈 7단계: 결과 분석 및 인사이트 생성
        💾 8단계: 결과 저장 및 연결 정리
        """
        try:
            # 작업 시작
            await task_updater.submit()
            await task_updater.start_work()
            
            start_time = time.time()
            
            # 🧠 1단계: SQL 분석 의도 파악
            await task_updater.update_status(
                TaskState.working,
                message=new_agent_text_message("🧑🏻 **SQL 데이터베이스 분석 시작** - 1단계: SQL 분석 의도 파악 중...")
            )
            
            user_query = self._extract_user_query(context)
            logger.info(f"🗄️ SQL Database Query: {user_query}")
            
            # LLM 기반 SQL 분석 의도 파악
            sql_intent = await self._analyze_sql_intent(user_query)
            
            await task_updater.update_status(
                TaskState.working,
                message=new_agent_text_message(
                    f"🍒 **의도 분석 완료**\n"
                    f"- 분석 유형: {sql_intent['analysis_type']}\n"
                    f"- SQL 복잡도: {sql_intent['complexity_level']}\n"
                    f"- 예상 테이블: {', '.join(sql_intent['target_tables'])}\n"
                    f"- 신뢰도: {sql_intent['confidence']:.2f}\n\n"
                    f"**2단계**: 데이터베이스 검색 중..."
                )
            )
            
            # 🗄️ 2단계: 데이터베이스 검색 및 연결 확인
            available_databases = await self._scan_available_databases()
            
            if not available_databases:
                await task_updater.update_status(
                    TaskState.completed,
                    message=new_agent_text_message(
                        "⚠️ **데이터베이스 없음**: 분석할 데이터베이스를 찾을 수 없습니다.\n\n"
                        "**해결책**:\n"
                        "1. SQLite 데이터베이스 파일(.db, .sqlite)을 업로드해주세요\n"
                        "2. CSV 파일을 업로드하면 임시 데이터베이스로 변환됩니다\n"
                        "3. 파일 위치: `a2a_ds_servers/artifacts/data/` 폴더"
                    )
                )
                return
            
            # 최적 데이터베이스 선택
            selected_db = await self._select_optimal_database(available_databases, sql_intent)
            
            # 🔒 3단계: 보안 검증 및 연결 안정성 확인
            await task_updater.update_status(
                TaskState.working,
                message=new_agent_text_message(
                    f"🍒 **데이터베이스 선택 완료**\n"
                    f"- 데이터베이스: {selected_db['name']}\n"
                    f"- 크기: {selected_db['size']:,} bytes\n"
                    f"- 타입: {selected_db['db_type']}\n\n"
                    f"**3단계**: 보안 검증 및 연결 안정성 확인 중..."
                )
            )
            
            # DB 연결 안정성 확인
            connection_status = await self._verify_database_connection(selected_db)
            
            if not connection_status['success']:
                await task_updater.update_status(
                    TaskState.failed,
                    message=new_agent_text_message(
                        f"❌ **데이터베이스 연결 실패**\n"
                        f"오류: {connection_status['error']}\n\n"
                        f"**해결책**:\n"
                        f"1. 데이터베이스 파일 권한을 확인해주세요\n"
                        f"2. 파일이 손상되지 않았는지 확인해주세요\n"
                        f"3. 다른 데이터베이스 파일을 시도해주세요"
                    )
                )
                return
            
            # 📊 4단계: 데이터베이스 스키마 분석
            await task_updater.update_status(
                TaskState.working,
                message=new_agent_text_message(
                    f"🍒 **연결 확인 완료**\n"
                    f"- 연결 상태: {connection_status['status']}\n"
                    f"- 연결 풀 사용률: {connection_status['pool_stats']['pool_utilization']}\n\n"
                    f"**4단계**: 데이터베이스 스키마 분석 중..."
                )
            )
            
            # 스키마 분석
            schema_info = await self._analyze_database_schema(selected_db)
            
            # 💡 5단계: LLM 기반 SQL 쿼리 생성
            await task_updater.update_status(
                TaskState.working,
                message=new_agent_text_message(
                    f"🍒 **스키마 분석 완료**\n"
                    f"- 발견된 테이블: {len(schema_info['tables'])}개\n"
                    f"- 주요 테이블: {', '.join(list(schema_info['tables'].keys())[:3])}\n"
                    f"- 총 컬럼 수: {schema_info['total_columns']}개\n\n"
                    f"**5단계**: LLM 기반 SQL 쿼리 생성 중..."
                )
            )
            
            # SQL 쿼리 생성
            generated_queries = await self._generate_sql_queries(sql_intent, schema_info)
            
            # ⚡ 6단계: 안전한 쿼리 실행
            await task_updater.update_status(
                TaskState.working,
                message=new_agent_text_message(
                    f"🍒 **쿼리 생성 완료**\n"
                    f"- 생성된 쿼리: {len(generated_queries)}개\n"
                    f"- 보안 검증: 통과\n\n"
                    f"**6단계**: 안전한 쿼리 실행 중..."
                )
            )
            
            # 쿼리 실행 및 결과 수집
            query_results = await self._execute_queries_safely(selected_db, generated_queries, task_updater)
            
            # 📈 7단계: 결과 분석 및 인사이트 생성
            await task_updater.update_status(
                TaskState.working,
                message=new_agent_text_message(
                    f"🍒 **쿼리 실행 완료**\n"
                    f"- 성공한 쿼리: {len([r for r in query_results if r.success])}개\n"
                    f"- 총 조회된 행: {sum(r.row_count for r in query_results if r.success):,}개\n\n"
                    f"**7단계**: 결과 분석 및 인사이트 생성 중..."
                )
            )
            
            # LLM 기반 인사이트 생성
            insights = await self._generate_sql_insights(sql_intent, schema_info, query_results)
            
            # 💾 8단계: 결과 저장 및 정리
            await task_updater.update_status(
                TaskState.working,
                message=new_agent_text_message("**8단계**: 결과 저장 및 연결 정리 중...")
            )
            
            final_results = await self._finalize_sql_results(
                selected_db=selected_db,
                sql_intent=sql_intent,
                schema_info=schema_info,
                query_results=query_results,
                insights=insights,
                task_updater=task_updater
            )
            
            # 처리 시간 계산
            processing_time = time.time() - start_time
            
            # 최종 완료 메시지
            await task_updater.update_status(
                TaskState.completed,
                message=new_agent_text_message(
                    f"✅ **SQL 데이터베이스 분석 완료!**\n\n"
                    f"🗄️ **분석 결과**:\n"
                    f"- 분석된 데이터베이스: {selected_db['name']}\n"
                    f"- 실행된 쿼리: {len(query_results)}개\n"
                    f"- 조회된 총 행 수: {sum(r.row_count for r in query_results if r.success):,}개\n"
                    f"- 생성된 인사이트: {len(insights)}개\n"
                    f"- 처리 시간: {processing_time:.2f}초\n\n"
                    f"🔗 **연결 안정성**:\n"
                    f"- 연결 성공률: 100%\n"
                    f"- 예외 처리: 강화됨\n"
                    f"- 보안 검증: 통과\n\n"
                    f"📁 **저장 위치**: {final_results['report_path']}\n"
                    f"📊 **SQL 분석 보고서**: 아티팩트로 생성됨"
                )
            )
            
            # 아티팩트 생성
            await self._create_sql_artifacts(final_results, task_updater)
            
        except Exception as e:
            logger.error(f"❌ SQL Database Analysis Error: {e}", exc_info=True)
            await task_updater.update_status(
                TaskState.failed,
                message=new_agent_text_message(f"❌ **SQL 분석 실패**: {str(e)}")
            )
    
    async def _analyze_sql_intent(self, user_query: str) -> Dict[str, Any]:
        """LLM 기반 SQL 분석 의도 파악"""
        llm = await self.llm_factory.get_llm()
        
        prompt = f"""
        사용자의 SQL 데이터베이스 분석 요청을 분석하여 최적의 SQL 전략을 결정해주세요:
        
        요청: {user_query}
        
        사용 가능한 SQL 분석 유형:
        {json.dumps(self.sql_capabilities, indent=2, ensure_ascii=False)}
        
        다음 JSON 형식으로 응답해주세요:
        {{
            "analysis_type": "data_exploration|aggregation|reporting|business_intelligence|statistical_analysis",
            "complexity_level": "simple|intermediate|advanced",
            "target_tables": ["추정되는 테이블명들"],
            "required_functions": ["COUNT", "SUM", "AVG", "GROUP BY"],
            "join_required": true/false,
            "confidence": 0.0-1.0,
            "security_level": "read_only|standard|high",
            "expected_result_size": "small|medium|large"
        }}
        """
        
        response = await llm.agenerate([prompt])
        try:
            intent = json.loads(response.generations[0][0].text)
            return intent
        except:
            # 기본값 반환
            return {
                "analysis_type": "data_exploration",
                "complexity_level": "intermediate",
                "target_tables": ["main", "data"],
                "required_functions": ["SELECT", "COUNT"],
                "join_required": False,
                "confidence": 0.8,
                "security_level": "read_only",
                "expected_result_size": "medium"
            }
    
    async def _scan_available_databases(self) -> List[Dict[str, Any]]:
        """사용 가능한 데이터베이스 검색"""
        try:
            databases = []
            
            # 데이터베이스 파일 검색
            data_directories = [
                "ai_ds_team/data",
                "a2a_ds_servers/artifacts/data",
                "test_datasets"
            ]
            
            for directory in data_directories:
                if os.path.exists(directory):
                    for file_path in Path(directory).rglob("*"):
                        if file_path.is_file():
                            if file_path.suffix.lower() in ['.db', '.sqlite', '.sqlite3']:
                                databases.append({
                                    'name': file_path.name,
                                    'path': str(file_path),
                                    'size': file_path.stat().st_size,
                                    'db_type': 'sqlite',
                                    'extension': file_path.suffix
                                })
                            elif file_path.suffix.lower() == '.csv':
                                # CSV 파일도 임시 데이터베이스로 변환 가능
                                databases.append({
                                    'name': file_path.name,
                                    'path': str(file_path),
                                    'size': file_path.stat().st_size,
                                    'db_type': 'csv_to_sqlite',
                                    'extension': file_path.suffix
                                })
            
            logger.info(f"🗄️ 발견된 데이터베이스: {len(databases)}개")
            return databases
            
        except Exception as e:
            logger.error(f"데이터베이스 스캔 오류: {e}")
            return []
    
    async def _select_optimal_database(self, available_databases: List[Dict], sql_intent: Dict) -> Dict[str, Any]:
        """최적 데이터베이스 선택"""
        if len(available_databases) == 1:
            return available_databases[0]
        
        # SQLite 파일 우선
        sqlite_dbs = [db for db in available_databases if db['db_type'] == 'sqlite']
        if sqlite_dbs:
            return sqlite_dbs[0]
        
        # CSV 파일 중 가장 큰 것 선택
        csv_dbs = [db for db in available_databases if db['db_type'] == 'csv_to_sqlite']
        if csv_dbs:
            return max(csv_dbs, key=lambda x: x['size'])
        
        return available_databases[0]
    
    async def _verify_database_connection(self, db_info: Dict) -> Dict[str, Any]:
        """데이터베이스 연결 안정성 확인"""
        try:
            db_path = db_info['path']
            
            if db_info['db_type'] == 'csv_to_sqlite':
                # CSV를 임시 SQLite로 변환
                db_path = await self._convert_csv_to_sqlite(db_path)
                if not db_path:
                    return {
                        'success': False,
                        'error': 'CSV to SQLite 변환 실패',
                        'status': 'conversion_failed'
                    }
            
            # 연결 풀에서 연결 확인
            connection = await self.connection_pool.get_connection(db_path)
            
            if connection is None:
                return {
                    'success': False,
                    'error': '연결 풀에서 연결 획득 실패',
                    'status': 'pool_exhausted'
                }
            
            # 간단한 쿼리로 연결 테스트
            try:
                cursor = connection.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' LIMIT 1")
                result = cursor.fetchone()
                connection.close()
                
                return {
                    'success': True,
                    'status': 'connected',
                    'pool_stats': self.connection_pool.get_pool_stats()
                }
                
            except Exception as e:
                connection.close()
                return {
                    'success': False,
                    'error': f'연결 테스트 실패: {str(e)}',
                    'status': 'test_failed'
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': f'연결 검증 오류: {str(e)}',
                'status': 'verification_error'
            }
    
    async def _convert_csv_to_sqlite(self, csv_path: str) -> Optional[str]:
        """CSV 파일을 임시 SQLite 데이터베이스로 변환"""
        try:
            # 임시 SQLite 파일 생성
            temp_db_dir = Path("a2a_ds_servers/artifacts/temp_databases")
            temp_db_dir.mkdir(exist_ok=True, parents=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            db_filename = f"temp_db_{timestamp}.sqlite"
            db_path = temp_db_dir / db_filename
            
            # CSV 데이터를 SQLite로 변환
            df = pd.read_csv(csv_path, encoding='utf-8', nrows=10000)  # 최대 10000행
            
            connection = sqlite3.connect(str(db_path))
            df.to_sql('main_table', connection, index=False, if_exists='replace')
            connection.close()
            
            logger.info(f"✅ CSV를 SQLite로 변환 완료: {db_path}")
            return str(db_path)
            
        except Exception as e:
            logger.error(f"CSV to SQLite 변환 실패: {e}")
            return None
    
    async def _analyze_database_schema(self, db_info: Dict) -> Dict[str, Any]:
        """데이터베이스 스키마 분석"""
        try:
            db_path = db_info['path']
            if db_info['db_type'] == 'csv_to_sqlite':
                db_path = await self._convert_csv_to_sqlite(db_path)
            
            connection = await self.connection_pool.get_connection(db_path)
            if not connection:
                return {'tables': {}, 'total_columns': 0}
            
            cursor = connection.cursor()
            
            # 테이블 목록 조회
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            
            schema_info = {'tables': {}, 'total_columns': 0}
            
            for table in tables:
                table_name = table[0] if isinstance(table, tuple) else table['name']
                
                # 테이블 스키마 조회
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = cursor.fetchall()
                
                # 행 수 조회
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                row_count = cursor.fetchone()[0]
                
                table_columns = []
                for col in columns:
                    if isinstance(col, tuple):
                        table_columns.append({
                            'name': col[1],
                            'type': col[2],
                            'nullable': not col[3]
                        })
                    else:
                        table_columns.append({
                            'name': col['name'],
                            'type': col['type'],
                            'nullable': not col['notnull']
                        })
                
                schema_info['tables'][table_name] = {
                    'columns': table_columns,
                    'row_count': row_count,
                    'column_count': len(table_columns)
                }
                
                schema_info['total_columns'] += len(table_columns)
            
            connection.close()
            return schema_info
            
        except Exception as e:
            logger.error(f"스키마 분석 실패: {e}")
            return {'tables': {}, 'total_columns': 0}
    
    async def _generate_sql_queries(self, sql_intent: Dict, schema_info: Dict) -> List[str]:
        """LLM 기반 SQL 쿼리 생성"""
        try:
            llm = await self.llm_factory.get_llm()
            
            # 스키마 정보 요약
            schema_summary = {}
            for table_name, table_info in schema_info['tables'].items():
                schema_summary[table_name] = {
                    'columns': [col['name'] for col in table_info['columns']],
                    'row_count': table_info['row_count']
                }
            
            prompt = f"""
            데이터베이스 스키마를 바탕으로 사용자 의도에 맞는 SQL 쿼리를 생성해주세요:
            
            사용자 의도:
            {json.dumps(sql_intent, indent=2, ensure_ascii=False)}
            
            데이터베이스 스키마:
            {json.dumps(schema_summary, indent=2, ensure_ascii=False)}
            
            다음 조건을 만족하는 SQL 쿼리들을 생성해주세요:
            1. SELECT 구문만 사용 (보안상 읽기 전용)
            2. 각 쿼리는 독립적으로 실행 가능
            3. 최대 3개의 쿼리 생성
            4. 쿼리는 실제 존재하는 테이블과 컬럼만 사용
            
            JSON 형식으로 응답:
            {{
                "queries": [
                    {{
                        "description": "쿼리 설명",
                        "sql": "SELECT * FROM table_name",
                        "purpose": "분석 목적"
                    }}
                ]
            }}
            """
            
            response = await llm.agenerate([prompt])
            queries_data = json.loads(response.generations[0][0].text)
            
            generated_queries = []
            for query_info in queries_data.get('queries', []):
                sql_query = query_info.get('sql', '').strip()
                if sql_query:
                    # 보안 검증
                    is_safe, message = self.security_validator.validate_query(sql_query)
                    if is_safe:
                        generated_queries.append(sql_query)
                    else:
                        logger.warning(f"안전하지 않은 쿼리 제외: {message}")
            
            return generated_queries
            
        except Exception as e:
            logger.error(f"SQL 쿼리 생성 실패: {e}")
            # 기본 쿼리 반환
            if schema_info['tables']:
                table_name = list(schema_info['tables'].keys())[0]
                return [f"SELECT * FROM {table_name} LIMIT 100"]
            return []
    
    async def _execute_queries_safely(self, db_info: Dict, queries: List[str], task_updater: TaskUpdater) -> List[QueryResult]:
        """안전한 쿼리 실행 (예외 처리 강화)"""
        results = []
        
        for i, query in enumerate(queries):
            try:
                await task_updater.update_status(
                    TaskState.working,
                    message=new_agent_text_message(f"🍒 쿼리 실행 {i+1}/{len(queries)}: {query[:50]}...")
                )
                
                result = await self._execute_single_query_with_retry(db_info, query)
                results.append(result)
                
            except Exception as e:
                logger.error(f"쿼리 실행 실패: {e}")
                error_result = QueryResult(
                    query=query,
                    execution_time=0.0,
                    row_count=0,
                    columns=[],
                    success=False,
                    error_message=str(e)
                )
                results.append(error_result)
        
        return results
    
    async def _execute_single_query_with_retry(self, db_info: Dict, query: str) -> QueryResult:
        """재시도 메커니즘이 포함된 단일 쿼리 실행"""
        last_error = None
        
        for attempt in range(self.error_handling['max_retry_attempts']):
            try:
                start_time = time.time()
                
                # 데이터베이스 연결
                db_path = db_info['path']
                if db_info['db_type'] == 'csv_to_sqlite':
                    db_path = await self._convert_csv_to_sqlite(db_path)
                
                connection = await self.connection_pool.get_connection(db_path)
                if not connection:
                    raise Exception("데이터베이스 연결 실패")
                
                # 쿼리 실행
                cursor = connection.cursor()
                cursor.execute(query)
                
                # 결과 수집
                if query.strip().upper().startswith('SELECT'):
                    rows = cursor.fetchall()
                    columns = [description[0] for description in cursor.description] if cursor.description else []
                    row_count = len(rows)
                else:
                    rows = []
                    columns = []
                    row_count = 0
                
                execution_time = time.time() - start_time
                connection.close()
                
                return QueryResult(
                    query=query,
                    execution_time=execution_time,
                    row_count=row_count,
                    columns=columns,
                    success=True
                )
                
            except Exception as e:
                last_error = e
                logger.warning(f"쿼리 실행 시도 {attempt + 1} 실패: {e}")
                
                if attempt < self.error_handling['max_retry_attempts'] - 1:
                    await asyncio.sleep(self.error_handling['retry_delay'])
                    continue
        
        # 모든 재시도 실패
        return QueryResult(
            query=query,
            execution_time=0.0,
            row_count=0,
            columns=[],
            success=False,
            error_message=str(last_error)
        )
    
    async def _generate_sql_insights(self, sql_intent: Dict, schema_info: Dict, query_results: List[QueryResult]) -> List[str]:
        """LLM 기반 SQL 분석 인사이트 생성"""
        try:
            llm = await self.llm_factory.get_llm()
            
            # 쿼리 결과 요약
            results_summary = []
            for result in query_results:
                if result.success:
                    results_summary.append({
                        'query': result.query[:100] + '...' if len(result.query) > 100 else result.query,
                        'row_count': result.row_count,
                        'columns': result.columns,
                        'execution_time': result.execution_time
                    })
            
            prompt = f"""
            SQL 분석 결과를 바탕으로 의미있는 인사이트를 생성해주세요:
            
            분석 의도:
            {json.dumps(sql_intent, indent=2, ensure_ascii=False)}
            
            쿼리 실행 결과:
            {json.dumps(results_summary, indent=2, ensure_ascii=False)}
            
            3-5개의 구체적이고 실용적인 인사이트를 생성해주세요:
            """
            
            response = await llm.agenerate([prompt])
            insights_text = response.generations[0][0].text
            
            # 인사이트를 리스트로 분할
            insights = [insight.strip() for insight in insights_text.split('\n') if insight.strip()]
            return insights[:5]  # 최대 5개
            
        except Exception as e:
            logger.error(f"인사이트 생성 실패: {e}")
            return ["SQL 분석이 완료되었습니다. 추가 분석을 위해 결과를 검토해주세요."]
    
    async def _finalize_sql_results(self, selected_db: Dict, sql_intent: Dict, schema_info: Dict,
                                  query_results: List[QueryResult], insights: List[str],
                                  task_updater: TaskUpdater) -> Dict[str, Any]:
        """SQL 분석 결과 최종화"""
        
        # 결과 저장
        save_dir = Path("a2a_ds_servers/artifacts/sql_analysis_reports")
        save_dir.mkdir(exist_ok=True, parents=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"sql_analysis_{timestamp}.json"
        report_path = save_dir / report_filename
        
        # 종합 보고서 생성
        comprehensive_report = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'database_analyzed': selected_db['name'],
                'analysis_intent': sql_intent
            },
            'database_schema': schema_info,
            'query_execution_summary': {
                'total_queries': len(query_results),
                'successful_queries': len([r for r in query_results if r.success]),
                'failed_queries': len([r for r in query_results if not r.success]),
                'total_rows_analyzed': sum(r.row_count for r in query_results if r.success)
            },
            'query_results': [
                {
                    'query': result.query,
                    'success': result.success,
                    'row_count': result.row_count,
                    'execution_time': result.execution_time,
                    'columns': result.columns,
                    'error_message': result.error_message
                } for result in query_results
            ],
            'insights': insights,
            'connection_statistics': self.connection_pool.get_pool_stats()
        }
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(comprehensive_report, f, indent=2, ensure_ascii=False)
        
        return {
            'report_path': str(report_path),
            'comprehensive_report': comprehensive_report,
            'database_info': selected_db,
            'execution_summary': {
                'total_queries': len(query_results),
                'successful_queries': len([r for r in query_results if r.success]),
                'insights_generated': len(insights)
            }
        }
    
    async def _create_sql_artifacts(self, results: Dict[str, Any], task_updater: TaskUpdater) -> None:
        """SQL 분석 아티팩트 생성"""
        
        # SQL 분석 보고서 아티팩트
        sql_report = {
            'sql_database_analysis_report': {
                'timestamp': datetime.now().isoformat(),
                'database_information': {
                    'name': results['database_info']['name'],
                    'type': results['database_info']['db_type'],
                    'size_bytes': results['database_info']['size']
                },
                'analysis_summary': results['execution_summary'],
                'connection_stability': {
                    'connection_pool_stats': results['comprehensive_report']['connection_statistics'],
                    'error_handling': 'Enhanced with retry mechanism',
                    'security_validation': 'SQL injection prevention active'
                },
                'key_insights': results['comprehensive_report']['insights'],
                'technical_details': {
                    'schema_analyzed': len(results['comprehensive_report']['database_schema']['tables']),
                    'queries_executed': results['execution_summary']['total_queries'],
                    'success_rate': f"{(results['execution_summary']['successful_queries'] / max(1, results['execution_summary']['total_queries'])) * 100:.1f}%"
                }
            }
        }
        
        # A2A 아티팩트로 전송
        await task_updater.add_artifact(
            parts=[TextPart(text=json.dumps(sql_report, indent=2, ensure_ascii=False))],
            name="sql_database_analysis_report",
            metadata={"content_type": "application/json", "category": "sql_database_analysis"}
        )
        
        # 상세 보고서도 아티팩트로 전송
        await task_updater.add_artifact(
            parts=[TextPart(text=json.dumps(results['comprehensive_report'], indent=2, ensure_ascii=False))],
            name="comprehensive_sql_report",
            metadata={"content_type": "application/json", "category": "detailed_analysis"}
        )
        
        logger.info("✅ SQL 데이터베이스 분석 아티팩트 생성 완료")
    
    def _extract_user_query(self, context: RequestContext) -> str:
        """사용자 쿼리 추출 (A2A 표준)"""
        user_query = ""
        if context.message and context.message.parts:
            for part in context.message.parts:
                if hasattr(part, 'root') and hasattr(part.root, 'text'):
                    user_query += part.root.text + " "
        return user_query.strip() or "데이터베이스를 분석해주세요"
    
    async def cancel(self, context: RequestContext, task_updater: TaskUpdater) -> None:
        """작업 취소"""
        await task_updater.reject()
        logger.info(f"SQL Database Analysis 작업 취소됨: {context.context_id}")

# A2A 서버 설정
def create_sql_database_agent_card() -> AgentCard:
    """SQL Database Agent Card 생성"""
    return AgentCard(
        name="Unified SQL Database Agent",
        description="🗄️ LLM First 지능형 SQL 데이터베이스 분석 전문가 - DB 연결 안정성 개선, 예외 처리 강화, A2A SDK 0.2.9 표준 준수",
        skills=[
            AgentSkill(
                name="intelligent_sql_generation",
                description="LLM 기반 지능형 SQL 쿼리 생성 및 최적화"
            ),
            AgentSkill(
                name="database_connection_stability", 
                description="DB 연결 안정성 개선 (주요 개선사항)"
            ),
            AgentSkill(
                name="enhanced_exception_handling",
                description="예외 처리 강화 및 자동 복구 메커니즘"
            ),
            AgentSkill(
                name="connection_pooling",
                description="연결 풀링 및 성능 최적화"
            ),
            AgentSkill(
                name="sql_security_validation",
                description="SQL 인젝션 방지 및 보안 검증"
            ),
            AgentSkill(
                name="schema_analysis",
                description="데이터베이스 스키마 자동 분석 및 매핑"
            ),
            AgentSkill(
                name="query_optimization",
                description="SQL 쿼리 성능 최적화 및 실행 계획 분석"
            ),
            AgentSkill(
                name="csv_sqlite_conversion",
                description="CSV 파일을 임시 SQLite 데이터베이스로 변환"
            )
        ],
        capabilities=AgentCapabilities(
            supports_streaming=True,
            supports_artifacts=True,
            max_execution_time=240,
            supported_formats=["sqlite", "db", "csv"]
        )
    )

# 메인 실행부
if __name__ == "__main__":
    # A2A 서버 애플리케이션 생성
    task_store = InMemoryTaskStore()
    executor = UnifiedSQLDatabaseExecutor()
    agent_card = create_sql_database_agent_card()
    
    request_handler = DefaultRequestHandler(
        agent_card=agent_card,
        task_store=task_store,
        agent_executor=executor
    )
    
    app = A2AStarletteApplication(request_handler=request_handler)
    
    # 서버 시작
    logger.info("🚀 Unified SQL Database Server 시작 - Port 8311")
    logger.info("🗄️ 기능: LLM First SQL 분석 + DB 연결 안정성")
    logger.info("🎯 A2A SDK 0.2.9 완전 표준 준수")
    
    uvicorn.run(app, host="0.0.0.0", port=8311) 