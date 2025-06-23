# 🦙 Ollama Tool Calling 완전 가이드
*CherryAI의 Ollama 통합 개선 및 설정 가이드*

## 📋 개요

이 가이드는 CherryAI에서 Ollama를 사용하여 GPT와 동일한 수준의 도구 호출(Tool Calling) 성능을 구현하는 방법을 설명합니다.

### ✅ 완료된 개선사항

- ✅ **패키지 호환성** - `langchain_ollama` 우선 사용으로 완전한 도구 호출 지원
- ✅ **모델 검증** - 도구 호출 지원 모델 자동 감지 및 권장
- ✅ **사용자 정의 Agent** - Ollama 전용 고도화된 도구 호출 에이전트
- ✅ **UI 강화** - 실시간 상태 모니터링 및 설정 가이드
- ✅ **자동 설정** - 환경변수 및 모델 자동 구성 스크립트
- ✅ **에러 처리** - 강화된 재시도 메커니즘 및 폴백 처리

## 🚀 빠른 시작

### 1. 자동 설정 (권장)

```bash
# Linux/macOS
./setup_ollama_env.sh

# Windows
setup_ollama_env.bat
```

### 2. 수동 설정

```bash
# 1. 패키지 설치
uv add ollama psutil

# 2. 환경변수 설정
export LLM_PROVIDER=OLLAMA
export OLLAMA_MODEL=llama3.1:8b
export OLLAMA_BASE_URL=http://localhost:11434
export OLLAMA_TIMEOUT=600

# 3. Ollama 서버 시작
ollama serve

# 4. 권장 모델 다운로드
ollama pull llama3.1:8b
```

## 🔧 핵심 기술 개선사항

### 1. LLM Factory 강화 (`core/llm_factory.py`)

#### 🆕 패키지 호환성 자동 감지
```python
# langchain_ollama 우선 사용 (도구 호출 지원)
try:
    from langchain_ollama import ChatOllama
    OLLAMA_TOOL_CALLING_SUPPORTED = True
except ImportError:
    from langchain_community.chat_models.ollama import ChatOllama
    OLLAMA_TOOL_CALLING_SUPPORTED = False
```

#### 🎯 모델 호환성 매핑
```python
# 도구 호출 지원 모델 (2024년 12월 기준)
OLLAMA_TOOL_CALLING_MODELS = {
    "llama3.1:8b", "llama3.1:70b", "llama3.2:3b",
    "qwen2.5:7b", "qwen2.5:14b", "qwen2.5-coder:7b",
    "mistral:7b", "gemma2:9b", "phi3:14b"
}

# 미지원 모델
OLLAMA_NON_TOOL_CALLING_MODELS = {
    "llama2", "qwen3:8b", "vicuna", "alpaca"
}
```

#### 🔍 자동 모델 추천
```python
def get_model_recommendation(ram_gb: Optional[int] = None) -> Dict[str, Any]:
    """사용자 시스템에 맞는 모델 추천"""
    if ram_gb >= 16:
        return {"name": "qwen2.5:14b", "description": "고성능 작업용"}
    elif ram_gb >= 10:
        return {"name": "llama3.1:8b", "description": "균형잡힌 성능"}
    else:
        return {"name": "qwen2.5:3b", "description": "가벼운 작업용"}
```

### 2. 사용자 정의 Ollama Agent (`app.py`)

#### 🤖 고도화된 도구 호출 처리
```python
def custom_ollama_agent(state):
    """Ollama용 고도화된 커스텀 에이전트"""
    
    # Enhanced prompting
    enhanced_prompt = """You are a data analysis expert.
    IMPORTANT: ALWAYS use available tools for data operations.
    Available tools: {tool_names}"""
    
    # Retry mechanism with tool enforcement
    max_retries = 3
    for attempt in range(max_retries):
        response = llm_with_tools.invoke(enhanced_messages)
        
        # Tool call processing
        if hasattr(response, 'tool_calls') and response.tool_calls:
            # Execute tools and get results
            tool_messages = []
            for tool_call in response.tool_calls:
                result = execute_tool(tool_call)
                tool_messages.append(result)
            
            # Generate final response with tool results
            final_response = llm.invoke(messages + [response] + tool_messages)
            return {"messages": [final_response]}
```

### 3. UI 모니터링 강화 (`ui/sidebar_components.py`)

#### 📊 실시간 Ollama 상태 표시
```python
def render_ollama_status():
    """고도화된 Ollama 상태 모니터링"""
    status = get_ollama_status()
    
    # 연결 상태, 패키지 정보, 모델 목록
    # 권장사항, 설정 제안 등 포함
```

## 🎯 권장 모델 목록

### 💪 고성능 시스템 (16GB+ RAM)
```bash
ollama pull qwen2.5:14b        # 고급 분석, 복잡한 코딩
ollama pull llama3.1:70b       # 최고 성능 (40GB RAM)
```

### ⚖️ 균형 시스템 (10-16GB RAM)
```bash
ollama pull llama3.1:8b        # 권장 기본 모델
ollama pull qwen2.5:7b         # 빠른 처리
ollama pull mistral:7b         # 우수한 추론
```

### 🪶 경량 시스템 (6-10GB RAM)
```bash
ollama pull qwen2.5:3b         # 가벼운 작업용
ollama pull llama3.2:3b        # Meta 최신 경량 모델
```

### 💻 코딩 전문
```bash
ollama pull qwen2.5-coder:7b   # 코딩 전문 모델
ollama pull codellama:7b       # Meta 코드 모델
```

## 🛠️ 고급 설정

### 환경변수 세부 설정
```bash
# 기본 설정
export LLM_PROVIDER=OLLAMA
export OLLAMA_MODEL=llama3.1:8b
export OLLAMA_BASE_URL=http://localhost:11434

# 성능 최적화
export OLLAMA_TIMEOUT=600              # 10분 타임아웃
export OLLAMA_AUTO_SWITCH_MODEL=true   # 자동 모델 전환

# 고급 설정
export OLLAMA_HOST=0.0.0.0             # 외부 접근 허용
export OLLAMA_PORT=11434               # 포트 설정
export OLLAMA_ORIGINS=*                # CORS 설정
```

### Ollama 서버 최적화
```bash
# GPU 메모리 설정
export OLLAMA_GPU_LAYERS=32

# 동시 처리 수 설정
export OLLAMA_MAX_LOADED_MODELS=2

# 메모리 제한
export OLLAMA_MAX_VRAM=8GB
```

## 🔍 문제 해결

### 1. 도구 호출이 작동하지 않을 때

#### 문제 진단
```python
from core.llm_factory import validate_llm_config, get_ollama_status

# 설정 검증
config = validate_llm_config()
print(config)

# Ollama 상태 확인
status = get_ollama_status()
print(status)
```

#### 해결 방법
1. **패키지 확인**: `langchain_ollama` 설치 확인
2. **모델 확인**: 도구 호출 지원 모델 사용
3. **연결 확인**: Ollama 서버 실행 상태
4. **버전 확인**: 최신 Ollama 버전 사용

### 2. 성능 최적화

#### 메모리 부족 시
```bash
# 경량 모델로 전환
export OLLAMA_MODEL=qwen2.5:3b

# 또는 초경량 모델
export OLLAMA_MODEL=qwen2.5:0.5b
```

#### 응답 속도 개선
```bash
# GPU 가속 활용
ollama serve --gpu

# 모델 사전 로딩
ollama run llama3.1:8b ""
```

### 3. 일반적인 오류 해결

#### Connection Refused
```bash
# Ollama 서버 시작
ollama serve

# 포트 확인
netstat -an | grep 11434
```

#### Model Not Found
```bash
# 사용 가능한 모델 확인
ollama list

# 모델 다운로드
ollama pull llama3.1:8b
```

#### Tool Calling Not Working
```bash
# 패키지 재설치
uv remove langchain-ollama
uv add langchain-ollama

# 모델 변경
export OLLAMA_MODEL=llama3.1:8b
```

## 📊 성능 비교

### GPT vs Ollama (도구 호출 성능)

| 작업 유형 | GPT-4 | Ollama (llama3.1:8b) | Ollama (qwen2.5:7b) |
|-----------|-------|----------------------|---------------------|
| 데이터 로딩 | ✅ 완벽 | ✅ 완벽 | ✅ 완벽 |
| 통계 분석 | ✅ 완벽 | ✅ 완벽 | ✅ 완벽 |
| 시각화 | ✅ 완벽 | ✅ 양호 | ✅ 양호 |
| 복잡 추론 | ✅ 완벽 | ✅ 양호 | ⚠️ 제한적 |
| 속도 | 🐌 보통 | 🐌 느림 | 🐇 빠름 |
| 비용 | 💰 유료 | 🆓 무료 | 🆓 무료 |

## 🚀 향후 개선 계획

### 단기 (1-2주)
- [ ] 모델별 성능 벤치마크 자동화
- [ ] 도구 사용량 통계 및 최적화
- [ ] 에러 복구 자동화 강화

### 중기 (1-2개월)
- [ ] 하이브리드 모드 (GPT + Ollama)
- [ ] 모델 앙상블 기능
- [ ] 자동 모델 스케일링

### 장기 (3-6개월)
- [ ] 커스텀 모델 파인튜닝 지원
- [ ] 분산 Ollama 클러스터 지원
- [ ] AI 모델 성능 예측 시스템

## 📞 지원 및 문의

- **이슈 리포트**: GitHub Issues
- **기능 요청**: GitHub Discussions
- **문서 개선**: Pull Request 환영

## 📚 참고 자료

- [Ollama 공식 문서](https://ollama.ai/docs)
- [LangChain Ollama 통합](https://python.langchain.com/docs/integrations/chat/ollama)
- [Tool Calling 가이드](https://docs.langchain.com/docs/modules/agents/tools/)

---

*이 가이드는 CherryAI v0.6.2+ 기준으로 작성되었습니다.*
*최신 업데이트는 [여기](./OLLAMA_IMPROVEMENT_GUIDE.md)에서 확인하세요.* 