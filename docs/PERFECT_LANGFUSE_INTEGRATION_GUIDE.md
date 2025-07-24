# 🎯 완벽한 Langfuse 통합 가이드

**작성일**: 2025-01-23  
**검증 완료**: DataCleaningAgent (Port 8306)  
**상태**: ✅ **100% 완성**

---

## 🏆 개요

DataCleaningAgent에서 **완벽한 Langfuse 통합**을 달성했습니다. 이 가이드는 다른 모든 에이전트에 동일한 수준의 추적 시스템을 적용하기 위한 **완전한 구현 방법론**입니다.

### 🎉 달성된 결과
- ✅ **null 값 완전 제거**: 모든 Input/Output이 의미있는 데이터
- ✅ **완전한 trace 구조**: 메인 트레이스 → 세부 span들
- ✅ **단계별 상세 추적**: 파싱 → 처리 → 저장의 전체 흐름
- ✅ **구조화된 데이터**: JSON 형태의 readable한 정보
- ✅ **오류 없는 안정성**: 모든 Langfuse API 호출 성공

---

## 📋 핵심 구현 패턴

### 1. 환경 설정 및 초기화

```python
# 1. 필수 임포트
import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# 2. Langfuse 통합 모듈 임포트
try:
    from core.universal_engine.langfuse_integration import SessionBasedTracer, LangfuseEnhancedA2AExecutor
    LANGFUSE_AVAILABLE = True
    logger.info("✅ Langfuse 통합 모듈 로드 성공")
except ImportError as e:
    LANGFUSE_AVAILABLE = False
    logger.warning(f"⚠️ Langfuse 통합 모듈 로드 실패: {e}")
```

### 2. AgentExecutor 클래스 초기화

```python
class YourAgentExecutor(AgentExecutor):
    """Langfuse 통합이 포함된 Agent Executor"""
    
    def __init__(self):
        """초기화"""
        # 기존 초기화 코드...
        
        # Langfuse 통합 초기화
        self.langfuse_tracer = None
        if LANGFUSE_AVAILABLE:
            try:
                self.langfuse_tracer = SessionBasedTracer()
                if self.langfuse_tracer.langfuse:
                    logger.info("✅ YourAgent Langfuse 통합 완료")
                else:
                    logger.warning("⚠️ Langfuse 설정 누락 - 기본 모드로 실행")
            except Exception as e:
                logger.error(f"❌ Langfuse 초기화 실패: {e}")
                self.langfuse_tracer = None
```

### 3. 메인 트레이스 생성 (핵심!)

```python
async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
    """에이전트 실행 (Langfuse 통합)"""
    
    # Langfuse 메인 트레이스 시작
    main_trace = None
    if self.langfuse_tracer and self.langfuse_tracer.langfuse:
        try:
            # 전체 사용자 쿼리 추출
            full_user_query = ""
            if context.message and hasattr(context.message, 'parts') and context.message.parts:
                for part in context.message.parts:
                    if hasattr(part, 'root') and part.root.kind == "text":
                        full_user_query += part.root.text + " "
                    elif hasattr(part, 'text'):
                        full_user_query += part.text + " "
            full_user_query = full_user_query.strip()
            
            # 메인 트레이스 생성 (task_id를 트레이스 ID로 사용)
            main_trace = self.langfuse_tracer.langfuse.trace(
                id=context.task_id,
                name="YourAgent_Execution",  # 에이전트명_Execution
                input=full_user_query,
                user_id="2055186",
                metadata={
                    "agent": "YourAgentName",
                    "port": YOUR_PORT,
                    "context_id": context.context_id,
                    "timestamp": str(context.task_id)
                }
            )
            logger.info(f"📊 Langfuse 메인 트레이스 시작: {context.task_id}")
        except Exception as e:
            logger.warning(f"⚠️ Langfuse 트레이스 생성 실패: {e}")
```

### 4. 단계별 Span 추가

#### A. 데이터 파싱 Span
```python
# 1단계: 데이터 파싱 (Langfuse 추적)
parsing_span = None
if main_trace:
    parsing_span = self.langfuse_tracer.langfuse.span(
        trace_id=context.task_id,
        name="data_parsing",
        input={"user_instructions": user_instructions[:500]},
        metadata={"step": "1", "description": "Parse data from user message"}
    )

logger.info("🔍 데이터 파싱 시작")
# 실제 파싱 로직...
parsed_data = your_parsing_logic(user_instructions)

# 파싱 결과 업데이트
if parsing_span:
    if parsed_data is not None:
        parsing_span.update(
            output={
                "success": True,
                "data_shape": list(parsed_data.shape),  # tuple을 list로 변환
                "columns": list(parsed_data.columns),
                "data_preview": parsed_data.head(3).to_dict('records'),
                "total_rows": len(parsed_data),
                "total_columns": len(parsed_data.columns)
            }
        )
    else:
        parsing_span.update(
            output={
                "success": False, 
                "reason": "No data found in message",
                "fallback_needed": True
            }
        )
```

#### B. 실제 처리 Span
```python
# 2단계: 실제 에이전트 처리 (Langfuse 추적)
processing_span = None
if main_trace:
    processing_span = self.langfuse_tracer.langfuse.span(
        trace_id=context.task_id,
        name="agent_processing",  # 에이전트별 맞춤 이름
        input={
            "input_data_shape": parsed_data.shape,
            "columns": list(parsed_data.columns),
            "user_instructions": user_instructions[:200]
        },
        metadata={"step": "2", "description": "Process data with agent"}
    )

logger.info("🚀 에이전트 처리 시작")
# 실제 처리 로직...
processing_results = your_agent_logic(parsed_data, user_instructions)

# 처리 결과 업데이트
if processing_span:
    processing_span.update(
        output={
            "success": True,
            "processed_data_shape": list(processing_results['data'].shape),
            "quality_score": processing_results.get('quality_score', 0),
            "operations_performed": len(processing_results.get('operations', [])),
            "processing_summary": processing_results.get('summary', [])[:3],
            "execution_time": processing_results.get('execution_time', 0)
        }
    )
```

#### C. 결과 저장 Span
```python
# 3단계: 결과 저장 (Langfuse 추적)
save_span = None
if main_trace:
    save_span = self.langfuse_tracer.langfuse.span(
        trace_id=context.task_id,
        name="save_results",
        input={
            "result_data_shape": processing_results['data'].shape,
            "quality_score": processing_results.get('quality_score', 0),
            "operations_count": len(processing_results.get('operations', []))
        },
        metadata={"step": "3", "description": "Save processed results to file"}
    )

# 파일 저장 로직...
output_path = f"path/to/results_{context.task_id}.csv"
processing_results['data'].to_csv(output_path, index=False)

# 저장 결과 업데이트
if save_span:
    save_span.update(
        output={
            "file_path": output_path,
            "file_size_mb": os.path.getsize(output_path) / (1024*1024),
            "saved_rows": len(processing_results['data']),
            "saved_successfully": True
        }
    )
```

### 5. 메인 트레이스 완료

#### A. 성공 시
```python
# 최종 응답 생성
result = generate_response(processing_results, user_instructions, output_path)

# A2A 응답
await task_updater.update_status(
    TaskState.completed,
    message=new_agent_text_message(result)
)

# Langfuse 메인 트레이스 완료
if main_trace:
    try:
        # Output을 요약된 형태로 제공
        output_summary = {
            "status": "completed",
            "result_preview": result[:1000] + "..." if len(result) > 1000 else result,
            "full_result_length": len(result)
        }
        
        main_trace.update(
            output=output_summary,
            metadata={
                "status": "completed",
                "result_length": len(result),
                "success": True,
                "completion_timestamp": str(context.task_id),
                "agent": "YourAgentName",
                "port": YOUR_PORT
            }
        )
        logger.info(f"📊 Langfuse 트레이스 완료: {context.task_id}")
    except Exception as e:
        logger.warning(f"⚠️ Langfuse 트레이스 완료 실패: {e}")
```

#### B. 오류 시
```python
except Exception as e:
    logger.error(f"❌ YourAgent 실행 오류: {e}")
    
    # Langfuse 메인 트레이스 오류 기록
    if main_trace:
        try:
            main_trace.update(
                output=f"Error: {str(e)}",
                metadata={
                    "status": "failed",
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "success": False,
                    "agent": "YourAgentName",
                    "port": YOUR_PORT
                }
            )
        except Exception as langfuse_error:
            logger.warning(f"⚠️ Langfuse 오류 기록 실패: {langfuse_error}")
    
    await task_updater.update_status(
        TaskState.failed,
        message=new_agent_text_message(f"처리 중 오류 발생: {str(e)}")
    )
```

---

## 🔧 중요한 구현 세부사항

### 1. 데이터 타입 변환
```python
# ❌ 잘못된 방식 (Langfuse에서 오류 발생)
"data_shape": df.shape,  # tuple

# ✅ 올바른 방식
"data_shape": list(df.shape),  # list
```

### 2. 문자열 길이 제한
```python
# 긴 텍스트는 잘라서 제공
"user_instructions": user_instructions[:500],
"result_preview": result[:1000] + "..." if len(result) > 1000 else result
```

### 3. 안전한 딕셔너리 접근
```python
# 안전한 데이터 추출
"quality_score": processing_results.get('quality_score', 0),
"operations_performed": len(processing_results.get('operations', [])),
```

### 4. 예외 처리
```python
# 모든 Langfuse 호출을 try-catch로 감싸기
try:
    span.update(output=data)
except Exception as e:
    logger.warning(f"⚠️ Langfuse 업데이트 실패: {e}")
```

---

## 📊 환경 변수 설정

`.env` 파일에 다음 설정이 필요합니다:

```bash
# Langfuse 설정
LANGFUSE_PUBLIC_KEY=pk-lf-xxxxxxxxxxxxxxxxx
LANGFUSE_SECRET_KEY=sk-lf-xxxxxxxxxxxxxxxxx
LANGFUSE_HOST=http://mangugil.synology.me:3001

# 기본 사용자 ID
EMP_NO=2055186
```

---

## 🎯 각 에이전트별 맞춤 설정

### DataVisualizationAgent
```python
main_trace = self.langfuse_tracer.langfuse.trace(
    name="DataVisualizationAgent_Execution",
    # ... 
)

# Span 이름들
"data_parsing"        # 데이터 파싱
"chart_generation"    # 차트 생성
"save_visualization"  # 시각화 저장
```

### EDAAgent
```python
main_trace = self.langfuse_tracer.langfuse.trace(
    name="EDAAgent_Execution",
    # ...
)

# Span 이름들
"data_parsing"     # 데이터 파싱
"eda_analysis"     # 탐색적 분석
"save_report"      # 리포트 저장
```

### FeatureEngineeringAgent
```python
main_trace = self.langfuse_tracer.langfuse.trace(
    name="FeatureEngineeringAgent_Execution",
    # ...
)

# Span 이름들
"data_parsing"           # 데이터 파싱
"feature_engineering"    # 피처 엔지니어링
"save_features"          # 피처 저장
```

---

## ✅ 검증 체크리스트

다른 에이전트에 적용 후 다음을 확인하세요:

### 1. 로그 확인
- [ ] 서버 시작 시 "✅ [Agent] Langfuse 통합 완료" 메시지
- [ ] "📊 Langfuse 메인 트레이스 시작" 메시지
- [ ] "📊 Langfuse 트레이스 완료" 메시지
- [ ] Langfuse 관련 오류 없음

### 2. Langfuse UI 확인
- [ ] **메인 트레이스**: 
  - Input: 전체 사용자 요청 (null 아님)
  - Output: 구조화된 결과 요약 (null 아님)
- [ ] **각 Span**: 
  - Input: 단계별 입력 정보 (null 아님)
  - Output: 단계별 결과 정보 (null 아님)
  - Metadata: 단계 설명
- [ ] **Trace ID**: task_id와 일치
- [ ] **User ID**: 2055186
- [ ] **타임스탬프**: 정확한 실행 시간

### 3. 기능 테스트
- [ ] 에이전트 정상 동작 (기존 기능 손상 없음)
- [ ] 복잡한 요청도 완전히 추적
- [ ] 오류 발생 시에도 trace 기록
- [ ] 응답 속도 영향 없음

---

## 🚀 다음 적용 우선순위

1. **DataVisualizationAgent** (Port 8308)
2. **EDAAgent** (Port 8312) 
3. **FeatureEngineeringAgent** (Port 8313)
4. **MLFlowAgent** (Port 8314)
5. **기타 모든 에이전트**

각 에이전트마다 이 가이드를 참고하여 **동일한 수준의 완벽한 Langfuse 통합**을 달성하세요!

---

**📋 작성자**: Claude  
**🎯 목표**: 모든 에이전트에서 완벽한 Langfuse 추적 달성  
**✅ 검증**: DataCleaningAgent 100% 완료  
**📅 업데이트**: 2025-01-23
