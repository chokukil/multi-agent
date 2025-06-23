# 🦙 Ollama 모델 도구 호출 개선 가이드

## 🚨 문제 상황
Ollama를 사용할 때 GPT와 다르게 **도구를 사용하지 않고 태스크를 조기에 종료**하는 문제가 발생했습니다.

## 🔍 근본 원인 분석

### 1️⃣ **모델별 도구 호출 능력 차이**
- **GPT-4**: OpenAI Function Calling을 완벽 지원
- **Ollama 모델**: 모델에 따라 도구 호출 능력이 제한적

### 2️⃣ **조기 종료 패턴**
- 시스템이 `"TASK COMPLETED:"` 문자열만으로 작업 완료 판단
- 실제 도구 사용 여부를 확인하지 않음

### 3️⃣ **모델별 처리 로직 부재**
- LLM 팩토리에서 모든 모델을 동일하게 처리
- 도구 호출 능력이 제한적인 모델에 대한 특별 처리 없음

## ✅ 구현된 해결책

### 1️⃣ **LLM 팩토리 개선** (`core/llm_factory.py`)

#### **도구 호출 능력 매핑**
```python
OLLAMA_TOOL_CALLING_MODELS = {
    "qwen2.5:7b", "qwen2.5:14b", "qwen2.5:32b", "qwen2.5:72b",
    "qwen3:8b", "qwen3:14b", "qwen3:32b", 
    "llama3.1:8b", "llama3.1:70b", "llama3.1:405b",
    "mistral:7b", "mixtral:8x7b", "gemma2:9b", "phi3:3.8b",
    # ... 기타 지원 모델들
}
```

#### **모델별 메타데이터 추가**
```python
# LLM 인스턴스에 능력 정보 추가
llm._tool_calling_capable = tool_calling_capable
llm._provider = "OLLAMA"
llm._model_name = model
llm._needs_enhanced_prompting = not tool_calling_capable
```

### 2️⃣ **Executor 강화** (`core/plan_execute/executor.py`)

#### **도구 사용 필요성 판단**
```python
def should_use_tools_for_task(task_type: str, task_description: str) -> bool:
    tool_required_tasks = {"eda", "analysis", "preprocessing", "visualization", "stats", "ml"}
    tool_keywords = ["데이터", "분석", "시각화", "통계", "그래프", "차트", "plot"]
    # 작업 유형과 키워드 기반 판단
```

#### **조기 완료 감지 및 방지**
```python
def detect_premature_completion(response_content: str, tools_used: bool, task_needs_tools: bool) -> bool:
    # "TASK COMPLETED"가 있지만 필요한 도구를 사용하지 않았는지 확인
    # 가설적 또는 예시 결과 제공 패턴 감지
```

#### **도구 호출 강제 프롬프팅**
```python
def create_enhanced_prompt_for_limited_models(task_prompt: str, tools_available: list) -> str:
    # 도구 사용을 강제하는 상세한 지시사항 생성
    # 금지 행동 명시 및 올바른 도구 사용법 안내
```

### 3️⃣ **UI 개선** (`ui/sidebar_components.py`)

#### **LLM 상태 표시 기능**
```python
def render_llm_status():
    # 현재 모델의 도구 호출 능력 표시
    # 권장 모델 목록 제공
    # 설정 가이드 제공
```

## 🎯 권장 Ollama 모델

### ✅ **도구 호출 지원 모델** (권장)
- **Qwen 시리즈**: `qwen2.5:7b`, `qwen3:8b`
- **Llama 시리즈**: `llama3.1:8b`, `llama3.2:3b`
- **Mistral 시리즈**: `mistral:7b`, `mixtral:8x7b`
- **기타**: `gemma2:9b`, `phi3:3.8b`

### ⚠️ **제한적 지원 모델**
- `llama2`와 같은 구형 모델들
- 시스템이 자동으로 강화된 프롬프팅 적용

## 🔧 설정 방법

### 1️⃣ **환경 변수 설정**
```bash
# 권장 모델 사용
export OLLAMA_MODEL=qwen2.5:7b
# 또는
export OLLAMA_MODEL=llama3.1:8b

# 타임아웃 증가 (Ollama는 로컬 처리로 더 오래 걸림)
export OLLAMA_TIMEOUT=600
```

### 2️⃣ **모델 다운로드**
```bash
# Ollama에서 권장 모델 다운로드
ollama pull qwen2.5:7b
ollama pull llama3.1:8b
ollama pull mistral:7b
```

## 📊 개선 효과

### **Before (문제 상황)**
```
📝 사용자: "데이터 기초통계 확인해줘"
🤖 Ollama: "데이터는 일반적으로 평균, 중앙값, 표준편차를 포함합니다..."
✅ TASK COMPLETED: 기초통계 개념을 설명했습니다. (도구 사용 없음)
```

### **After (개선 후)**
```
📝 사용자: "데이터 기초통계 확인해줘"
🤖 Ollama: 도구 사용 감지 → python_repl_ast 호출
🔧 python_repl_ast: df = get_current_data(); print(df.describe())
📊 실제 데이터 분석 결과 출력
✅ TASK COMPLETED: 실제 데이터의 기초통계를 분석했습니다.
```

## 🛡️ 보호 메커니즘

### 1️⃣ **조기 완료 방지**
- 도구 사용이 필요한 작업에서 도구를 사용하지 않으면 재시도 요구
- 가설적 결과 제공 시 경고 및 재지시

### 2️⃣ **프롬프트 강화**
- 도구 호출 능력이 제한적인 모델에 강화된 지시사항 제공
- 명확한 도구 사용 가이드라인 제시

### 3️⃣ **실시간 모니터링**
- UI에서 현재 모델의 도구 호출 능력 실시간 표시
- 권장 모델로의 전환 가이드 제공

## 🔮 향후 개선 계획

### 1️⃣ **더 많은 모델 지원**
- 새로운 Ollama 모델들의 도구 호출 능력 테스트 및 추가
- 커뮤니티 피드백을 통한 모델 리스트 확장

### 2️⃣ **적응형 프롬프팅**
- 모델의 응답 패턴을 학습하여 더 효과적인 프롬프트 생성
- 모델별 최적화된 지시사항 개발

### 3️⃣ **성능 모니터링**
- 모델별 도구 사용률 및 성공률 추적
- 자동 모델 추천 시스템 구축

## 📝 사용 팁

### ✅ **DO**
- 권장 모델 사용 (`qwen2.5:7b`, `llama3.1:8b` 등)
- UI에서 LLM 상태 확인
- 충분한 타임아웃 설정 (10분 이상)

### ❌ **DON'T**
- 구형 모델 (`llama2` 등) 단독 사용
- 너무 짧은 타임아웃 설정
- 도구 호출 능력 확인 없이 복잡한 분석 요청

---

이 가이드를 통해 Ollama 모델의 도구 호출 문제가 크게 개선되었습니다. 추가 문제가 발생하면 GitHub Issues에 보고해 주세요! 