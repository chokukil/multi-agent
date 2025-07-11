# CherryAI 환경 설정 가이드

## EMP_NO 설정
Langfuse에서 사용자 식별을 위해 EMP_NO 환경변수를 설정하세요.

### .env 파일 생성
프로젝트 루트에 .env 파일을 생성하고 다음 내용을 추가하세요:

```
# 직원 번호 설정 (Langfuse User ID로 사용)
EMP_NO=EMP001

# Langfuse 설정
LANGFUSE_PUBLIC_KEY=your_public_key_here
LANGFUSE_SECRET_KEY=your_secret_key_here
LANGFUSE_HOST=http://localhost:3000

# 로깅 설정
LOGGING_PROVIDER=langfuse

# OpenAI API 설정
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini

# LLM 설정
LLM_PROVIDER=OPENAI
LLM_TEMPERATURE=0.7
```

### 설정 완료 후
- Langfuse 대시보드에서 EMP_NO 값으로 사용자 활동을 추적할 수 있습니다
- 모든 AI 에이전트의 로그에 직원 번호가 기록됩니다

