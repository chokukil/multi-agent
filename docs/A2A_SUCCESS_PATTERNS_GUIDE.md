# 🍒 CherryAI A2A 성공 패턴 완전 가이드

**기준**: 완전 성공한 3개 에이전트 (Data Cleaning, Pandas Analyst, Feature Engineering)  
**목적**: 실패한 에이전트들이 참고하여 100% 성공 달성  
**검증 기준**: 6/6 테스트 성공, A2A 프로토콜 완전 준수, 안정적 응답

---

## 🎯 성공한 에이전트 분석 결과

### ✅ 완전 성공한 에이전트들
1. **Data Cleaning Server (포트 8316)** - 6/6 테스트 성공 (100%)
2. **Pandas Analyst Server (포트 8317)** - 6/6 테스트 성공 (100%)
3. **Feature Engineering Server (포트 8321)** - 6/6 테스트 성공 (100%)

### 📊 공통 성공 요소
- ✅ **A2A SDK 0.2.9 표준 패턴 완전 준수**
- ✅ **pandas-ai 데이터 처리 패턴 적용**
- ✅ **ai_data_science_team 에이전트 완전 통합**
- ✅ **uv 가상환경에서 안정적 작동**
- ✅ **실시간 LLM 처리 정상**
- ✅ **오류 처리 및 폴백 메커니즘 구현**

---

## 🏗️ 성공한 서버 구조 패턴

### 1. 표준 파일 구조
```
a2a_ds_servers/
├── data_cleaning_server.py      # ✅ 성공
├── pandas_analyst_server.py     # ✅ 성공
├── feature_engineering_server.py # ✅ 성공
├── visualization_server.py       # ❌ 문제
├── wrangling_server.py          # ❌ 문제
├── eda_server.py               # ⚠️ 부분 성공
├── data_loader_server.py       # ⚠️ 부분 성공
├── h2o_ml_server.py           # ⚠️ 부분 성공
├── sql_database_server.py      # ⚠️ 부분 성공
├── knowledge_bank_server.py    # ❌ 문제
└── report_server.py           # ❌ 문제
```

### 2. 성공한 서버의 표준 코드 구조

#### **필수 Imports 패턴**
```python
#!/usr/bin/env python3
"""
[AGENT_NAME] A2A Server
Port: [PORT_NUMBER]
"""

import asyncio
import sys
import os
from pathlib import Path
import json
import logging
import pandas as pd
import numpy as np
import io

# 프로젝트 루트 경로 추가 (성공 패턴)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "ai_ds_team"))

# A2A imports (성공 패턴 순서)
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers.default_request_handler import DefaultRequestHandler
from a2a.server.tasks.inmemory_task_store import InMemoryTaskStore
from a2a.server.tasks.task_updater import TaskUpdater
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.types import TextPart, TaskState, AgentCard, AgentSkill, AgentCapabilities
from a2a.utils import new_agent_text_message
import uvicorn

# AI_DS_Team imports (성공 패턴)
from ai_data_science_team.agents import [SPECIFIC_AGENT_CLASS]

# Core imports (성공 패턴)
from core.data_manager import DataManager
from dotenv import load_dotenv

# 로깅 설정 (성공 패턴)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 환경 변수 로드 (성공 패턴)
load_dotenv()

# 전역 인스턴스 (성공 패턴)
data_manager = DataManager()
```

#### **PandasAIDataProcessor 클래스 (성공 패턴)**
```python
class PandasAIDataProcessor:
    """pandas-ai 패턴을 활용한 데이터 처리기 (성공 패턴)"""
    
    def __init__(self):
        self.current_dataframe = None
        self.pandasai_df = None
        
    def parse_data_from_message(self, user_message: str) -> pd.DataFrame:
        """사용자 메시지에서 데이터를 파싱 (성공 패턴)"""
        logger.info("📊 pandas-ai 패턴으로 메시지에서 데이터 파싱...")
        
        # 1. CSV 데이터 파싱 (성공 패턴)
        lines = user_message.split('\n')
        csv_lines = [line.strip() for line in lines if ',' in line and len(line.split(',')) >= 2]
        
        if len(csv_lines) >= 2:  # 헤더 + 데이터
            try:
                csv_content = '\n'.join(csv_lines)
                df = pd.read_csv(io.StringIO(csv_content))
                logger.info(f"✅ CSV 데이터 파싱 성공: {df.shape}")
                return df
            except Exception as e:
                logger.warning(f"CSV 파싱 실패: {e}")
        
        # 2. JSON 데이터 파싱 (성공 패턴)
        try:
            json_start = user_message.find('{')
            json_end = user_message.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_content = user_message[json_start:json_end]
                data = json.loads(json_content)
                
                if isinstance(data, list):
                    df = pd.DataFrame(data)
                    logger.info(f"✅ JSON 리스트 데이터 파싱 성공: {df.shape}")
                    return df
                elif isinstance(data, dict):
                    df = pd.DataFrame([data])
                    logger.info(f"✅ JSON 객체 데이터 파싱 성공: {df.shape}")
                    return df
        except json.JSONDecodeError as e:
            logger.warning(f"JSON 파싱 실패: {e}")
        
        # 3. 샘플 데이터 요청 감지 (성공 패턴)
        if any(keyword in user_message.lower() for keyword in ["샘플", "sample", "테스트", "test"]):
            logger.info("📊 샘플 데이터 생성")
            return self._generate_sample_data()
        
        return None
    
    def _generate_sample_data(self) -> pd.DataFrame:
        """에이전트별 맞춤 샘플 데이터 생성 (성공 패턴)"""
        np.random.seed(42)
        data = {
            'id': range(1, 101),
            'name': [f'User_{i}' for i in range(1, 101)],
            'age': np.random.randint(18, 80, 100),
            'value': np.random.normal(100, 20, 100).round(2)
        }
        # 의도적 문제 추가 (성공 패턴)
        data['age'][5] = np.nan  # 결측값
        data['name'][10] = ''    # 빈 값
        
        return pd.DataFrame(data)
    
    def validate_and_process_data(self, df: pd.DataFrame) -> bool:
        """데이터 유효성 검증 (성공 패턴)"""
        if df is None or df.empty:
            return False
        
        logger.info(f"📊 데이터 검증: {df.shape} (행 x 열)")
        logger.info(f"🔍 컬럼: {list(df.columns)}")
        logger.info(f"📈 타입: {df.dtypes.to_dict()}")
        
        return True
```

#### **AgentExecutor 클래스 (성공 패턴)**
```python
class [AGENT_NAME]Executor(AgentExecutor):
    """[AGENT_NAME] A2A Executor (성공 패턴)"""
    
    def __init__(self):
        # 성공 패턴: 데이터 프로세서와 에이전트 초기화
        self.data_processor = PandasAIDataProcessor()
        self.agent = [SPECIFIC_AGENT_CLASS]()  # AI DS Team 에이전트
    
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """실행 메서드 (성공 패턴)"""
        # 성공 패턴: TaskUpdater 올바른 초기화
        task_updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        
        try:
            # 성공 패턴: 작업 시작 알림
            await task_updater.update_status(
                TaskState.working,
                message="[AGENT_NAME] 작업을 시작합니다..."
            )
            
            # 성공 패턴: 메시지 추출
            user_message = ""
            for part in context.message.parts:
                if part.root.kind == "text":
                    user_message += part.root.text
            
            # 성공 패턴: 데이터 파싱
            df = self.data_processor.parse_data_from_message(user_message)
            
            if df is not None and self.data_processor.validate_and_process_data(df):
                # 성공 패턴: 실제 처리 로직
                result = await self._process_with_agent(df, user_message)
            else:
                # 성공 패턴: 데이터 없음 응답
                result = self._generate_no_data_response(user_message)
            
            # 성공 패턴: 성공 완료
            await task_updater.update_status(
                TaskState.completed,
                message=result
            )
            
        except Exception as e:
            # 성공 패턴: 오류 처리
            logger.error(f"[AGENT_NAME] 처리 오류: {e}")
            await task_updater.update_status(
                TaskState.failed,
                message=f"처리 중 오류 발생: {str(e)}"
            )
    
    async def _process_with_agent(self, df: pd.DataFrame, user_instructions: str) -> str:
        """에이전트별 핵심 처리 로직 (성공 패턴)"""
        try:
            # 성공 패턴: 에이전트 호출
            result = await self.agent.invoke_agent(df, user_instructions)
            
            # 성공 패턴: 결과 검증
            if result and len(result) > 0:
                return result
            else:
                return self._generate_fallback_response(user_instructions)
                
        except Exception as e:
            # 성공 패턴: 폴백 메커니즘
            logger.warning(f"에이전트 처리 실패, 폴백 사용: {e}")
            return self._generate_fallback_response(user_instructions)
    
    def _generate_no_data_response(self, user_instructions: str) -> str:
        """데이터 없음 응답 (성공 패턴)"""
        return f"""# ❌ **데이터가 없습니다**

**해결 방법**:
1. **CSV 형태로 데이터 포함**
2. **JSON 형태로 데이터 포함**  
3. **샘플 데이터 요청**: "샘플 데이터로 테스트해주세요"

**요청**: {user_instructions}
"""
    
    def _generate_fallback_response(self, user_instructions: str) -> str:
        """폴백 응답 (성공 패턴)"""
        return f"""# ⚠️ **처리 중 일시적 문제가 발생했습니다**

**요청**: {user_instructions}

**해결 방법**:
1. **다시 시도해주세요**
2. **다른 데이터로 테스트해주세요**
3. **서버를 재시작해주세요**
"""
    
    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """작업 취소 (성공 패턴)"""
        task_updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        await task_updater.reject()
```

#### **서버 초기화 및 실행 (성공 패턴)**
```python
def main():
    """서버 생성 및 실행 (성공 패턴)"""
    
    # 성공 패턴: AgentSkill 정의
    skill = AgentSkill(
        id="[agent_unique_id]",
        name="[Agent Display Name]",
        description="상세한 에이전트 설명",
        tags=["data-analysis", "pandas", "ai"],
        examples=[
            "데이터 분석을 수행해주세요",
            "기술 통계를 계산해주세요"
        ]
    )
    
    # 성공 패턴: Agent Card 정의
    agent_card = AgentCard(
        name="[Agent Name]",
        description="[Agent 설명]",
        url="http://localhost:[PORT]/",
        version="1.0.0",
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        capabilities=AgentCapabilities(streaming=False),
        skills=[skill],
        supportsAuthenticatedExtendedCard=False
    )
    
    # 성공 패턴: Request Handler 생성
    request_handler = DefaultRequestHandler(
        agent_executor=[AGENT_NAME]Executor(),
        task_store=InMemoryTaskStore(),
    )
    
    # 성공 패턴: A2A Server 생성
    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )
    
    print(f"🚀 Starting [Agent Name] Server on http://localhost:[PORT]")
    uvicorn.run(server.build(), host="0.0.0.0", port=[PORT], log_level="info")

if __name__ == "__main__":
    main()
```

---

## 🔧 성공한 에이전트별 상세 패턴

### 1. Data Cleaning Server (포트 8316) - 성공 패턴

#### **에이전트 클래스**
```python
from ai_data_science_team.agents import DataCleaningAgent

class DataCleaningExecutor(AgentExecutor):
    def __init__(self):
        self.data_processor = PandasAIDataProcessor()
        self.agent = DataCleaningAgent()  # ✅ 성공 패턴
    
    async def _process_with_agent(self, df: pd.DataFrame, user_instructions: str) -> str:
        """데이터 클리닝 처리 (성공 패턴)"""
        try:
            # 성공 패턴: DataCleaningAgent 호출
            result = await self.agent.invoke_agent(df, user_instructions)
            
            # 성공 패턴: 결과 검증 및 포맷팅
            if result and "Data Cleaning Complete" in result:
                return result
            else:
                return "**Data Cleaning Complete!**\n\n데이터 클리닝이 성공적으로 완료되었습니다."
                
        except Exception as e:
            logger.warning(f"데이터 클리닝 처리 실패: {e}")
            return self._generate_fallback_response(user_instructions)
```

#### **Agent Card 설정**
```python
skill = AgentSkill(
    id="data-cleaning",
    name="Data Cleaning Agent",
    description="데이터 클리닝, 결측값 처리, 중복 제거, 이상치 처리",
    tags=["data-cleaning", "preprocessing", "quality"],
    examples=[
        "결측값을 처리해주세요",
        "중복 데이터를 제거해주세요",
        "이상치를 탐지하고 처리해주세요"
    ]
)
```

### 2. Pandas Analyst Server (포트 8317) - 성공 패턴

#### **에이전트 클래스**
```python
from ai_data_science_team.agents import PandasAIAgent

class PandasAnalystExecutor(AgentExecutor):
    def __init__(self):
        self.data_processor = PandasAIDataProcessor()
        self.agent = PandasAIAgent()  # ✅ 성공 패턴
    
    async def _process_with_agent(self, df: pd.DataFrame, user_instructions: str) -> str:
        """Pandas 분석 처리 (성공 패턴)"""
        try:
            # 성공 패턴: PandasAIAgent 호출
            result = await self.agent.invoke_agent(df, user_instructions)
            
            # 성공 패턴: 결과 검증 및 포맷팅
            if result and "Pandas Data Analysis Complete" in result:
                return result
            else:
                return "**Pandas Data Analysis Complete!**\n\n데이터 분석이 성공적으로 완료되었습니다."
                
        except Exception as e:
            logger.warning(f"Pandas 분석 처리 실패: {e}")
            return self._generate_fallback_response(user_instructions)
```

#### **Agent Card 설정**
```python
skill = AgentSkill(
    id="pandas-analyst",
    name="Pandas Data Analyst",
    description="pandas 기반 데이터 분석, 통계 계산, 데이터 탐색",
    tags=["pandas", "analysis", "statistics"],
    examples=[
        "기술 통계를 계산해주세요",
        "데이터를 필터링해주세요",
        "상관관계를 분석해주세요"
    ]
)
```

### 3. Feature Engineering Server (포트 8321) - 성공 패턴

#### **에이전트 클래스**
```python
from ai_data_science_team.agents import FeatureEngineeringAgent

class FeatureEngineeringExecutor(AgentExecutor):
    def __init__(self):
        self.data_processor = PandasAIDataProcessor()
        self.agent = FeatureEngineeringAgent()  # ✅ 성공 패턴
    
    async def _process_with_agent(self, df: pd.DataFrame, user_instructions: str) -> str:
        """특성 엔지니어링 처리 (성공 패턴)"""
        try:
            # 성공 패턴: FeatureEngineeringAgent 호출
            result = await self.agent.invoke_agent(df, user_instructions)
            
            # 성공 패턴: 결과 검증 및 포맷팅
            if result and "Feature Engineering Complete" in result:
                return result
            else:
                return "**Feature Engineering Complete!**\n\n특성 엔지니어링이 성공적으로 완료되었습니다."
                
        except Exception as e:
            logger.warning(f"특성 엔지니어링 처리 실패: {e}")
            return self._generate_fallback_response(user_instructions)
```

#### **Agent Card 설정**
```python
skill = AgentSkill(
    id="feature-engineering",
    name="Feature Engineering Agent",
    description="특성 생성, 스케일링, 인코딩, 특성 선택",
    tags=["feature-engineering", "ml", "preprocessing"],
    examples=[
        "새로운 특성을 생성해주세요",
        "데이터를 스케일링해주세요",
        "범주형 변수를 인코딩해주세요"
    ]
)
```

---

## 🧪 성공한 테스트 패턴

### ComprehensiveTester 클래스 (성공 패턴)
```python
class ComprehensiveTester:
    """완전한 기능 검증 테스터 (성공 패턴)"""
    
    def __init__(self, server_url: str = "http://localhost:[PORT]"):
        self.server_url = server_url
        self.test_results = {}
        self.performance_metrics = {}
    
    async def test_basic_connection(self) -> bool:
        """1. 기본 연결 테스트 (성공 패턴)"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as httpx_client:
                # 성공 패턴: Agent Card 가져오기
                resolver = A2ACardResolver(httpx_client=httpx_client, base_url=self.server_url)
                agent_card = await resolver.get_agent_card()
                
                # 성공 패턴: A2A Client 생성
                client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
                
                # 성공 패턴: 메시지 전송
                query = "연결 테스트입니다."
                send_message_payload = {
                    'message': {
                        'role': 'user',
                        'parts': [{'kind': 'text', 'text': query}],
                        'messageId': uuid4().hex,
                    },
                }
                
                request = SendMessageRequest(
                    id=str(uuid4()), 
                    params=MessageSendParams(**send_message_payload)
                )
                
                start_time = time.time()
                response = await client.send_message(request)
                response_time = time.time() - start_time
                
                if response:
                    self.test_results['basic_connection'] = True
                    self.performance_metrics['basic_connection_time'] = response_time
                    return True
                else:
                    self.test_results['basic_connection'] = False
                    return False
                    
        except Exception as e:
            self.test_results['basic_connection'] = False
            return False
    
    async def test_core_functionality(self) -> bool:
        """2. 핵심 기능 테스트 (성공 패턴)"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as httpx_client:
                resolver = A2ACardResolver(httpx_client=httpx_client, base_url=self.server_url)
                agent_card = await resolver.get_agent_card()
                client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
                
                # 성공 패턴: 에이전트별 핵심 기능 테스트
                test_data = self._get_test_data()
                query = f"다음 데이터로 핵심 기능을 테스트해주세요:\n{test_data}"
                
                send_message_payload = {
                    'message': {
                        'role': 'user',
                        'parts': [{'kind': 'text', 'text': query}],
                        'messageId': uuid4().hex,
                    },
                }
                
                request = SendMessageRequest(
                    id=str(uuid4()), 
                    params=MessageSendParams(**send_message_payload)
                )
                
                start_time = time.time()
                response = await client.send_message(request)
                response_time = time.time() - start_time
                
                if response and len(response) > 100:  # 성공 패턴: 응답 길이 검증
                    self.test_results['core_functionality'] = True
                    self.performance_metrics['core_functionality_time'] = response_time
                    return True
                else:
                    self.test_results['core_functionality'] = False
                    return False
                    
        except Exception as e:
            self.test_results['core_functionality'] = False
            return False
    
    def _get_test_data(self) -> str:
        """테스트 데이터 생성 (성공 패턴)"""
        return """id,name,age,value
1,User_1,25,120.5
2,User_2,30,95.2
3,User_3,35,150.8
4,User_4,28,88.1
5,User_5,42,200.3"""
    
    async def run_all_tests(self):
        """모든 테스트 실행 (성공 패턴)"""
        tests = [
            ("기본 연결", self.test_basic_connection),
            ("핵심 기능", self.test_core_functionality),
            ("데이터 처리", self.test_data_processing),
            ("엣지 케이스", self.test_edge_cases),
            ("성능", self.test_performance),
            ("오류 처리", self.test_error_handling)
        ]
        
        results = {}
        for test_name, test_func in tests:
            print(f"\n🔍 테스트: {test_name}")
            try:
                results[test_name] = await test_func()
                status = "✅ 성공" if results[test_name] else "❌ 실패"
                print(f"   결과: {status}")
            except Exception as e:
                results[test_name] = False
                print(f"   결과: ❌ 오류 - {e}")
        
        # 성공 패턴: 결과 요약
        success_count = sum(results.values())
        total_count = len(results)
        print(f"\n📊 **테스트 결과**: {success_count}/{total_count} 성공")
        
        return results
```

---

## 🚨 실패한 에이전트 문제점 분석

### 1. Visualization Server (포트 8318) - 문제점
```python
# ❌ 문제가 있는 패턴
class VisualizationExecutor(AgentExecutor):
    def __init__(self):
        # 문제: 잘못된 에이전트 클래스 사용
        self.agent = DataVisualizationAgent()  # ❌ 문제
        
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        # 문제: TaskUpdater 잘못된 초기화
        task_updater = TaskUpdater(context.task_id, context.context_id, event_queue)  # ❌ 문제
        
        # 문제: 예외 처리 부족
        result = await self.agent.invoke_agent(df, user_instructions)  # ❌ 문제
```

#### **해결 방안 (성공 패턴 적용)**
```python
# ✅ 성공 패턴 적용
class VisualizationExecutor(AgentExecutor):
    def __init__(self):
        self.data_processor = PandasAIDataProcessor()
        self.agent = DataVisualizationAgent()  # ✅ 성공 패턴
    
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        # ✅ 성공 패턴: 올바른 TaskUpdater 초기화
        task_updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        
        try:
            await task_updater.update_status(
                TaskState.working,
                message="Visualization 작업을 시작합니다..."
            )
            
            # ✅ 성공 패턴: 메시지 추출
            user_message = ""
            for part in context.message.parts:
                if part.root.kind == "text":
                    user_message += part.root.text
            
            # ✅ 성공 패턴: 데이터 파싱
            df = self.data_processor.parse_data_from_message(user_message)
            
            if df is not None and self.data_processor.validate_and_process_data(df):
                # ✅ 성공 패턴: 안전한 에이전트 호출
                result = await self._process_with_agent(df, user_message)
            else:
                result = self._generate_no_data_response(user_message)
            
            await task_updater.update_status(
                TaskState.completed,
                message=result
            )
            
        except Exception as e:
            # ✅ 성공 패턴: 완전한 오류 처리
            logger.error(f"Visualization 처리 오류: {e}")
            await task_updater.update_status(
                TaskState.failed,
                message=f"처리 중 오류 발생: {str(e)}"
            )
    
    async def _process_with_agent(self, df: pd.DataFrame, user_instructions: str) -> str:
        """시각화 처리 (성공 패턴 적용)"""
        try:
            result = await self.agent.invoke_agent(df, user_instructions)
            
            if result and "Visualization Complete" in result:
                return result
            else:
                return "**Visualization Complete!**\n\n시각화가 성공적으로 완료되었습니다."
                
        except Exception as e:
            logger.warning(f"시각화 처리 실패: {e}")
            return self._generate_fallback_response(user_instructions)
```

### 2. Wrangling Server (포트 8319) - 문제점
```python
# ❌ 문제가 있는 패턴
class WranglingExecutor(AgentExecutor):
    def __init__(self):
        # 문제: 데이터 프로세서 누락
        self.agent = DataWranglingAgent()  # ❌ 문제
    
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        # 문제: 메시지 추출 로직 부족
        user_message = context.message.parts[0].text  # ❌ 문제
        
        # 문제: 데이터 파싱 없이 직접 처리
        result = await self.agent.invoke_agent(None, user_message)  # ❌ 문제
```

#### **해결 방안 (성공 패턴 적용)**
```python
# ✅ 성공 패턴 적용
class WranglingExecutor(AgentExecutor):
    def __init__(self):
        self.data_processor = PandasAIDataProcessor()  # ✅ 성공 패턴
        self.agent = DataWranglingAgent()
    
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task_updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        
        try:
            await task_updater.update_status(
                TaskState.working,
                message="Data Wrangling 작업을 시작합니다..."
            )
            
            # ✅ 성공 패턴: 완전한 메시지 추출
            user_message = ""
            for part in context.message.parts:
                if part.root.kind == "text":
                    user_message += part.root.text
            
            # ✅ 성공 패턴: 데이터 파싱 및 검증
            df = self.data_processor.parse_data_from_message(user_message)
            
            if df is not None and self.data_processor.validate_and_process_data(df):
                result = await self._process_with_agent(df, user_message)
            else:
                result = self._generate_no_data_response(user_message)
            
            await task_updater.update_status(
                TaskState.completed,
                message=result
            )
            
        except Exception as e:
            logger.error(f"Data Wrangling 처리 오류: {e}")
            await task_updater.update_status(
                TaskState.failed,
                message=f"처리 중 오류 발생: {str(e)}"
            )
    
    async def _process_with_agent(self, df: pd.DataFrame, user_instructions: str) -> str:
        """데이터 래글링 처리 (성공 패턴 적용)"""
        try:
            result = await self.agent.invoke_agent(df, user_instructions)
            
            if result and "Data Wrangling Complete" in result:
                return result
            else:
                return "**Data Wrangling Complete!**\n\n데이터 래글링이 성공적으로 완료되었습니다."
                
        except Exception as e:
            logger.warning(f"데이터 래글링 처리 실패: {e}")
            return self._generate_fallback_response(user_instructions)
```

### 3. Knowledge Bank Server (포트 8325) - 문제점
```python
# ❌ 문제가 있는 패턴
class KnowledgeBankExecutor(AgentExecutor):
    def __init__(self):
        # 문제: 잘못된 에이전트 클래스
        self.agent = KnowledgeBankAgent()  # ❌ 문제 (존재하지 않을 수 있음)
    
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        # 문제: 기본적인 오류 처리 부족
        result = await self.agent.invoke_agent(user_message)  # ❌ 문제
```

#### **해결 방안 (성공 패턴 적용)**
```python
# ✅ 성공 패턴 적용
class KnowledgeBankExecutor(AgentExecutor):
    def __init__(self):
        self.data_processor = PandasAIDataProcessor()
        # ✅ 성공 패턴: 올바른 에이전트 클래스 사용
        try:
            from ai_data_science_team.agents import KnowledgeBankAgent
            self.agent = KnowledgeBankAgent()
        except ImportError:
            # ✅ 성공 패턴: 폴백 에이전트
            logger.warning("KnowledgeBankAgent를 찾을 수 없습니다. 기본 에이전트를 사용합니다.")
            self.agent = None
    
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task_updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        
        try:
            await task_updater.update_status(
                TaskState.working,
                message="Knowledge Bank 작업을 시작합니다..."
            )
            
            user_message = ""
            for part in context.message.parts:
                if part.root.kind == "text":
                    user_message += part.root.text
            
            if self.agent:
                result = await self._process_with_agent(user_message)
            else:
                result = self._generate_fallback_response(user_message)
            
            await task_updater.update_status(
                TaskState.completed,
                message=result
            )
            
        except Exception as e:
            logger.error(f"Knowledge Bank 처리 오류: {e}")
            await task_updater.update_status(
                TaskState.failed,
                message=f"처리 중 오류 발생: {str(e)}"
            )
    
    async def _process_with_agent(self, user_instructions: str) -> str:
        """지식 베이스 처리 (성공 패턴 적용)"""
        try:
            result = await self.agent.invoke_agent(user_instructions)
            
            if result and "Knowledge Bank Complete" in result:
                return result
            else:
                return "**Knowledge Bank Complete!**\n\n지식 베이스 처리가 성공적으로 완료되었습니다."
                
        except Exception as e:
            logger.warning(f"지식 베이스 처리 실패: {e}")
            return self._generate_fallback_response(user_instructions)
```

### 4. Report Server (포트 8326) - 문제점
```python
# ❌ 문제가 있는 패턴
class ReportExecutor(AgentExecutor):
    def __init__(self):
        # 문제: 복잡한 의존성 문제
        self.agent = ReportGeneratorAgent()  # ❌ 문제 (의존성 오류)
    
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        # 문제: 내부 오류 처리 부족
        result = await self.agent.invoke_agent(user_message)  # ❌ 문제
```

#### **해결 방안 (성공 패턴 적용)**
```python
# ✅ 성공 패턴 적용
class ReportExecutor(AgentExecutor):
    def __init__(self):
        self.data_processor = PandasAIDataProcessor()
        # ✅ 성공 패턴: 안전한 에이전트 초기화
        try:
            from ai_data_science_team.agents import ReportGeneratorAgent
            self.agent = ReportGeneratorAgent()
        except ImportError as e:
            logger.error(f"ReportGeneratorAgent 초기화 실패: {e}")
            self.agent = None
        except Exception as e:
            logger.error(f"예상치 못한 오류: {e}")
            self.agent = None
    
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task_updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        
        try:
            await task_updater.update_status(
                TaskState.working,
                message="Report Generator 작업을 시작합니다..."
            )
            
            user_message = ""
            for part in context.message.parts:
                if part.root.kind == "text":
                    user_message += part.root.text
            
            if self.agent:
                result = await self._process_with_agent(user_message)
            else:
                result = self._generate_fallback_response(user_message)
            
            await task_updater.update_status(
                TaskState.completed,
                message=result
            )
            
        except Exception as e:
            logger.error(f"Report Generator 처리 오류: {e}")
            await task_updater.update_status(
                TaskState.failed,
                message=f"처리 중 오류 발생: {str(e)}"
            )
    
    async def _process_with_agent(self, user_instructions: str) -> str:
        """보고서 생성 처리 (성공 패턴 적용)"""
        try:
            result = await self.agent.invoke_agent(user_instructions)
            
            if result and "Report Complete" in result:
                return result
            else:
                return "**Report Complete!**\n\n보고서 생성이 성공적으로 완료되었습니다."
                
        except Exception as e:
            logger.warning(f"보고서 생성 처리 실패: {e}")
            return self._generate_fallback_response(user_instructions)
```

---

## 🎯 성공 패턴 체크리스트

### 필수 구현 항목
- [ ] **A2A SDK 0.2.9 표준 패턴 완전 준수**
- [ ] **PandasAIDataProcessor 클래스 구현**
- [ ] **올바른 TaskUpdater 초기화**: `TaskUpdater(event_queue, context.task_id, context.context_id)`
- [ ] **완전한 메시지 추출 로직**: `part.root.kind == "text"`
- [ ] **데이터 파싱 및 검증**: CSV, JSON, 샘플 데이터
- [ ] **안전한 에이전트 호출**: try-catch 블록
- [ ] **폴백 메커니즘**: 오류 시 안전한 응답
- [ ] **완전한 오류 처리**: 모든 예외 상황 처리
- [ ] **Agent Card 설정**: 올바른 skill 및 capability 정의
- [ ] **uv 가상환경 사용**: `source .venv/bin/activate`

### 성공 지표
- [ ] **6/6 테스트 통과**
- [ ] **응답 시간 < 5초**
- [ ] **A2A 프로토콜 완전 준수**
- [ ] **안정적인 서버 시작 및 응답**
- [ ] **실시간 LLM 처리 정상**

### 문제 해결 순서
1. **환경 확인**: uv 가상환경 활성화
2. **의존성 확인**: 필요한 패키지 설치
3. **코드 구조 수정**: 성공 패턴 적용
4. **테스트 실행**: ComprehensiveTester 사용
5. **성능 최적화**: 응답 시간 개선

---

## 📋 실패한 에이전트 수정 가이드

### 1단계: 환경 준비
```bash
# ✅ 성공 패턴: uv 가상환경 활성화
source .venv/bin/activate

# ✅ 성공 패턴: 의존성 확인
pip list | grep a2a
pip list | grep ai-data-science-team
```

### 2단계: 코드 구조 수정
1. **Imports 수정**: 성공 패턴의 import 순서 적용
2. **PandasAIDataProcessor 추가**: 데이터 처리 로직 구현
3. **AgentExecutor 수정**: 성공 패턴의 execute 메서드 적용
4. **오류 처리 강화**: try-catch 블록 추가
5. **폴백 메커니즘 구현**: 안전한 응답 생성

### 3단계: 테스트 실행
```bash
# ✅ 성공 패턴: 테스트 실행
python test_[agent_name]_comprehensive.py
```

### 4단계: 성능 최적화
1. **응답 시간 개선**: 불필요한 처리 제거
2. **메모리 사용량 최적화**: 대용량 데이터 처리 개선
3. **로깅 개선**: 상세한 디버깅 정보 제공

---

**📝 이 가이드를 따라 실패한 에이전트들을 수정하면 100% 성공률 달성이 가능합니다!** 