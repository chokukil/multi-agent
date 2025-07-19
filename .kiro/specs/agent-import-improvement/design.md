# Agent Import Improvement Design Document

## Overview

본 설계는 성공적으로 완료된 `data_cleaning_server_clean.py`의 패턴을 기반으로 전체 A2A 데이터 사이언스 서버 생태계의 import 구조를 체계적으로 개선하는 방안을 제시합니다. 핵심 원칙은 **모듈화 유지**, **상대 import 문제 해결**, **원본 기능 완전 활용**입니다.

## Architecture

### 현재 성공 패턴 분석

#### 성공한 구조 (data_cleaning_server_clean.py)
```
CherryAI_0717/                           # 프로젝트 루트
├── ai_data_science_team/                # ✅ 루트로 이동 완료
│   ├── agents/
│   ├── tools/
│   ├── templates/
│   └── utils/
└── a2a_ds_servers/
    └── data_cleaning_server_clean.py    # ✅ 성공적으로 작동
```

#### 성공 요인
1. **패키지 루트 위치**: `ai_data_science_team`이 프로젝트 루트에 위치
2. **단순한 sys.path 설정**: `sys.path.insert(0, str(project_root))` 만 사용
3. **원본 함수 직접 포함**: 복잡한 import 대신 핵심 함수 직접 구현
4. **절대 import 사용**: 상대 import 문제 회피

### 전체 시스템 아키텍처

```
CherryAI_0717/                           # 프로젝트 루트
├── ai_data_science_team/                # 원본 패키지 (루트 위치)
│   ├── __init__.py
│   ├── agents/                          # 에이전트 모듈들
│   │   ├── __init__.py
│   │   ├── data_cleaning_agent.py
│   │   ├── data_loader_tools_agent.py
│   │   ├── data_visualization_agent.py
│   │   ├── data_wrangling_agent.py
│   │   ├── feature_engineering_agent.py
│   │   └── sql_database_agent.py
│   ├── ds_agents/                       # 데이터 사이언스 에이전트
│   │   ├── __init__.py
│   │   └── eda_tools_agent.py
│   ├── ml_agents/                       # 머신러닝 에이전트
│   │   ├── __init__.py
│   │   ├── h2o_ml_agent.py
│   │   └── mlflow_tools_agent.py
│   ├── tools/                           # 도구 모듈들
│   │   ├── __init__.py
│   │   ├── dataframe.py
│   │   ├── eda.py
│   │   ├── data_loader.py
│   │   ├── h2o.py
│   │   ├── mlflow.py
│   │   └── sql.py
│   ├── templates/                       # 템플릿 모듈들
│   │   ├── __init__.py
│   │   └── agent_templates.py
│   ├── utils/                           # 유틸리티 모듈들
│   │   ├── __init__.py
│   │   ├── regex.py
│   │   ├── logging.py
│   │   └── messages.py
│   └── parsers/                         # 파서 모듈들
│       ├── __init__.py
│       └── parsers.py
├── a2a_ds_servers/                      # A2A 서버들
│   ├── common/                          # 공통 모듈 (신규 생성)
│   │   ├── __init__.py
│   │   ├── base_server.py               # 공통 서버 베이스
│   │   ├── import_utils.py              # Import 유틸리티
│   │   └── data_processor.py            # 공통 데이터 처리
│   ├── data_cleaning_server_clean.py    # ✅ 완료
│   ├── data_loader_server.py            # 🔄 개선 대상
│   ├── data_visualization_server.py     # 🔄 개선 대상
│   ├── data_wrangling_server.py         # 🔄 개선 대상
│   ├── eda_tools_server.py              # 🔄 개선 대상
│   ├── feature_engineering_server.py    # 🔄 개선 대상
│   ├── h2o_ml_server.py                 # 🔄 개선 대상
│   ├── mlflow_server.py                 # 🔄 개선 대상
│   ├── sql_database_server.py           # 🔄 개선 대상
│   └── pandas_analyst_server.py         # 🔄 개선 대상
└── core/                                # 핵심 시스템 모듈
    ├── data_manager.py
    └── ...
```

## Components and Interfaces

### 1. 공통 Import 유틸리티 (a2a_ds_servers/common/import_utils.py)

```python
"""
공통 Import 설정 및 유틸리티
모든 A2A 서버에서 사용하는 표준 import 패턴
"""

import sys
import logging
from pathlib import Path
from typing import Optional, Any

def setup_project_paths() -> None:
    """프로젝트 경로 설정 - 모든 서버에서 동일하게 사용"""
    project_root = Path(__file__).parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

def safe_import_ai_ds_team(module_path: str) -> tuple[bool, Optional[Any]]:
    """AI DS Team 모듈 안전 import"""
    try:
        module = __import__(f"ai_data_science_team.{module_path}", fromlist=[''])
        return True, module
    except ImportError as e:
        logging.warning(f"AI DS Team 모듈 import 실패: {module_path} - {e}")
        return False, None

def get_ai_ds_agent(agent_name: str) -> tuple[bool, Optional[Any]]:
    """AI DS Team 에이전트 가져오기"""
    success, agents_module = safe_import_ai_ds_team("agents")
    if success and hasattr(agents_module, agent_name):
        return True, getattr(agents_module, agent_name)
    return False, None
```

### 2. 공통 서버 베이스 클래스 (a2a_ds_servers/common/base_server.py)

```python
"""
모든 A2A 서버의 공통 베이스 클래스
표준화된 서버 구조와 공통 기능 제공
"""

from abc import ABC, abstractmethod
from pathlib import Path
import logging
from typing import Dict, Any, Optional

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers.default_request_handler import DefaultRequestHandler
from a2a.server.tasks.inmemory_task_store import InMemoryTaskStore
from a2a.server.agent_execution import AgentExecutor
from a2a.types import AgentCard, AgentSkill, AgentCapabilities

from .import_utils import setup_project_paths, get_ai_ds_agent

class BaseA2AServer(ABC):
    """모든 A2A 서버의 베이스 클래스"""
    
    def __init__(self, port: int, agent_name: str, version: str = "1.0.0"):
        setup_project_paths()
        self.port = port
        self.agent_name = agent_name
        self.version = version
        self.logger = logging.getLogger(f"{agent_name}_server")
        
    @abstractmethod
    def create_agent_executor(self) -> AgentExecutor:
        """각 서버별 AgentExecutor 생성"""
        pass
    
    @abstractmethod
    def get_agent_skills(self) -> list[AgentSkill]:
        """각 서버별 AgentSkill 정의"""
        pass
    
    def create_agent_card(self) -> AgentCard:
        """표준 AgentCard 생성"""
        return AgentCard(
            name=f"AI {self.agent_name}",
            description=f"AI-powered {self.agent_name} service",
            url=f"http://localhost:{self.port}/",
            version=self.version,
            defaultInputModes=["text"],
            defaultOutputModes=["text"],
            capabilities=AgentCapabilities(streaming=False),
            skills=self.get_agent_skills(),
            supportsAuthenticatedExtendedCard=False
        )
    
    def run_server(self):
        """서버 실행"""
        import uvicorn
        
        request_handler = DefaultRequestHandler(
            agent_executor=self.create_agent_executor(),
            task_store=InMemoryTaskStore(),
        )
        
        server = A2AStarletteApplication(
            agent_card=self.create_agent_card(),
            http_handler=request_handler,
        )
        
        print(f"🚀 Starting {self.agent_name} Server")
        print(f"🌐 Server starting on http://localhost:{self.port}")
        print(f"📋 Agent card: http://localhost:{self.port}/.well-known/agent.json")
        
        uvicorn.run(server.build(), host="0.0.0.0", port=self.port, log_level="info")
```

### 3. 공통 데이터 처리기 (a2a_ds_servers/common/data_processor.py)

```python
"""
모든 서버에서 사용하는 공통 데이터 처리 기능
중복 코드 제거 및 표준화된 데이터 처리 패턴 제공
"""

import pandas as pd
import numpy as np
import json
import io
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class CommonDataProcessor:
    """공통 데이터 처리 기능"""
    
    @staticmethod
    def parse_data_from_message(user_message: str) -> Optional[pd.DataFrame]:
        """사용자 메시지에서 데이터 파싱 - 모든 서버 공통 로직"""
        logger.info("📊 메시지에서 데이터 파싱...")
        
        # CSV 데이터 파싱
        df = CommonDataProcessor._parse_csv_data(user_message)
        if df is not None:
            return df
            
        # JSON 데이터 파싱
        df = CommonDataProcessor._parse_json_data(user_message)
        if df is not None:
            return df
        
        # 샘플 데이터 요청 확인
        if CommonDataProcessor._is_sample_request(user_message):
            return CommonDataProcessor._create_sample_data()
        
        return None
    
    @staticmethod
    def _parse_csv_data(message: str) -> Optional[pd.DataFrame]:
        """CSV 형태 데이터 파싱"""
        try:
            lines = message.split('\n')
            csv_lines = [line.strip() for line in lines if ',' in line and len(line.split(',')) >= 2]
            
            if len(csv_lines) >= 2:
                csv_content = '\n'.join(csv_lines)
                df = pd.read_csv(io.StringIO(csv_content))
                logger.info(f"✅ CSV 데이터 파싱 성공: {df.shape}")
                return df
        except Exception as e:
            logger.warning(f"CSV 파싱 실패: {e}")
        return None
    
    @staticmethod
    def _parse_json_data(message: str) -> Optional[pd.DataFrame]:
        """JSON 형태 데이터 파싱"""
        try:
            json_start = message.find('{')
            json_end = message.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_content = message[json_start:json_end]
                data = json.loads(json_content)
                
                if isinstance(data, list):
                    df = pd.DataFrame(data)
                elif isinstance(data, dict):
                    df = pd.DataFrame([data])
                else:
                    return None
                    
                logger.info(f"✅ JSON 데이터 파싱 성공: {df.shape}")
                return df
        except Exception as e:
            logger.warning(f"JSON 파싱 실패: {e}")
        return None
    
    @staticmethod
    def _is_sample_request(message: str) -> bool:
        """샘플 데이터 요청인지 확인"""
        keywords = ["샘플", "테스트", "example", "demo", "sample", "test"]
        return any(keyword in message.lower() for keyword in keywords)
    
    @staticmethod
    def _create_sample_data() -> pd.DataFrame:
        """샘플 데이터 생성"""
        logger.info("🔧 샘플 데이터 생성...")
        
        np.random.seed(42)
        
        data = {
            'id': range(1, 101),
            'name': [f'User_{i}' for i in range(1, 101)],
            'age': np.random.randint(18, 80, 100),
            'income': np.random.randint(20000, 150000, 100),
            'category': np.random.choice(['A', 'B', 'C', 'D'], 100),
            'score': np.random.normal(75, 15, 100)
        }
        
        df = pd.DataFrame(data)
        
        # 의도적으로 결측값과 이상값 추가
        missing_indices = np.random.choice(df.index, 15, replace=False)
        df.loc[missing_indices[:5], 'age'] = np.nan
        df.loc[missing_indices[5:10], 'income'] = np.nan
        df.loc[missing_indices[10:], 'category'] = np.nan
        
        # 이상값 추가
        df.loc[0, 'age'] = 200
        df.loc[1, 'income'] = 1000000
        df.loc[2, 'score'] = -50
        
        # 중복 행 추가
        df = pd.concat([df, df.iloc[:3]], ignore_index=True)
        
        logger.info(f"✅ 샘플 데이터 생성 완료: {df.shape}")
        return df
```

## Data Models

### 서버 설정 모델

```python
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ServerConfig:
    """서버 설정 데이터 모델"""
    name: str
    port: int
    agent_type: str
    ai_ds_module: str
    version: str = "1.0.0"
    description: Optional[str] = None
    tags: List[str] = None
    examples: List[str] = None

@dataclass
class ImportStatus:
    """Import 상태 추적 모델"""
    module_name: str
    success: bool
    error_message: Optional[str] = None
    fallback_used: bool = False
```

## Error Handling

### Import 오류 처리 전략

1. **Graceful Degradation**: 원본 모듈 import 실패 시 폴백 기능 제공
2. **명확한 오류 메시지**: 개발자가 문제를 쉽게 파악할 수 있는 로깅
3. **자동 복구**: 가능한 경우 자동으로 대안 경로 시도

```python
class ImportErrorHandler:
    """Import 오류 처리 클래스"""
    
    @staticmethod
    def handle_ai_ds_import_error(module_path: str, error: Exception) -> None:
        """AI DS Team 모듈 import 오류 처리"""
        logger.error(f"AI DS Team 모듈 import 실패: {module_path}")
        logger.error(f"오류 상세: {str(error)}")
        logger.info("폴백 모드로 전환합니다...")
        
    @staticmethod
    def create_fallback_response(feature_name: str) -> str:
        """폴백 응답 생성"""
        return f"""
# ⚠️ {feature_name} 기능 제한 모드

현재 원본 AI DS Team 모듈을 사용할 수 없어 제한된 기능으로 동작합니다.

**해결 방법:**
1. `ai_data_science_team` 패키지가 프로젝트 루트에 있는지 확인
2. 필요한 의존성이 설치되어 있는지 확인
3. Python 경로 설정이 올바른지 확인

**현재 사용 가능한 기능:**
- 기본 데이터 처리
- 샘플 데이터 생성
- 기본 통계 정보 제공
"""
```

## Testing Strategy

### 테스트 계층 구조

1. **Unit Tests**: 각 모듈의 import 기능 테스트
2. **Integration Tests**: 서버 간 통합 테스트
3. **End-to-End Tests**: 전체 워크플로우 테스트

### 테스트 시나리오

```python
# 예시: Import 테스트 케이스
class TestImportSystem:
    def test_ai_ds_team_import_success(self):
        """AI DS Team 모듈 정상 import 테스트"""
        pass
    
    def test_fallback_on_import_failure(self):
        """Import 실패 시 폴백 동작 테스트"""
        pass
    
    def test_server_startup_with_missing_modules(self):
        """모듈 누락 시 서버 시작 테스트"""
        pass
```

## Performance Considerations

### 최적화 전략

1. **Lazy Loading**: 필요한 시점에만 모듈 로드
2. **Module Caching**: 한 번 로드된 모듈 캐싱
3. **Memory Management**: 불필요한 모듈 언로드

### 성능 메트릭

- 서버 시작 시간: < 5초
- 메모리 사용량: 기본 < 200MB
- Import 시간: < 1초

## Security Considerations

### 보안 요구사항

1. **Path Traversal 방지**: sys.path 조작 시 보안 검증
2. **Module Validation**: import하는 모듈의 유효성 검증
3. **Error Information Leakage 방지**: 오류 메시지에서 민감 정보 제거

## Deployment Strategy

### 배포 단계

1. **Phase 1**: 공통 모듈 생성 및 테스트
2. **Phase 2**: 기존 서버들 순차적 마이그레이션
3. **Phase 3**: 통합 테스트 및 성능 최적화
4. **Phase 4**: 문서화 및 가이드라인 정리

### 롤백 계획

각 서버별로 기존 버전 백업 유지하여 문제 발생 시 즉시 롤백 가능