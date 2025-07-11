# 🚀 CherryAI v2.0 설치 가이드

**완벽한 설치를 위한 단계별 가이드**

## 📋 시스템 요구사항

### 최소 요구사항
- **Python**: 3.12 이상
- **운영체제**: macOS, Linux, Windows 10/11
- **메모리**: 8GB RAM (16GB 권장)
- **저장공간**: 10GB 이상 여유 공간
- **네트워크**: 인터넷 연결 (API 키 및 패키지 설치)

### 권장 사양
- **Python**: 3.12.10 (최신 테스트 버전)
- **메모리**: 16GB+ RAM
- **CPU**: 4코어 이상 (8코어 권장)
- **저장공간**: SSD 권장

## 🛠️ 사전 준비

### 1. Python 설치 확인

```bash
python --version
# 출력 예시: Python 3.12.10
```

Python 3.12+ 미설치 시:

#### macOS
```bash
# Homebrew 사용
brew install python@3.12

# 또는 pyenv 사용
pyenv install 3.12.10
pyenv global 3.12.10
```

#### Ubuntu/Debian
```bash
sudo apt update
sudo apt install python3.12 python3.12-venv python3.12-pip
```

#### Windows
[Python 공식 사이트](https://python.org)에서 Python 3.12+ 다운로드 및 설치

### 2. UV 패키지 매니저 설치 (권장)

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# pip 사용
pip install uv
```

설치 확인:
```bash
uv --version
```

### 3. Git 설치 확인

```bash
git --version
```

## 📦 CherryAI 설치

### 1. 저장소 클론

```bash
# 저장소 클론
git clone <repository-url>
cd CherryAI_0623

# 또는 특정 버전
git clone -b v2.0 <repository-url>
cd CherryAI_0623
```

### 2. 가상환경 생성 및 활성화

#### UV 사용 (권장)
```bash
# 가상환경 생성
uv venv

# 가상환경 활성화
# macOS/Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

#### Venv 사용
```bash
# 가상환경 생성
python -m venv .venv

# 가상환경 활성화
# macOS/Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### 3. 의존성 설치

#### UV 사용 (권장)
```bash
# 모든 의존성 설치
uv pip install -e .

# 또는 개발용 의존성 포함
uv pip install -e ".[dev]"
```

#### Pip 사용
```bash
# 기본 설치
pip install -e .

# 또는 개발용 의존성 포함
pip install -e ".[dev]"
```

### 4. 설치 검증

```bash
# 핵심 모듈 임포트 테스트
python -c "
import streamlit
import pandas
import numpy
import a2a
print('✅ 모든 핵심 패키지가 설치되었습니다!')
"
```

## 🔑 환경 설정

### 1. .env 파일 생성

```bash
# 템플릿 복사
cp .env.example .env

# 편집기로 열기
nano .env  # 또는 code .env
```

### 2. 필수 환경 변수 설정

**.env 파일 예시:**
```env
# OpenAI API 설정
OPENAI_API_KEY=sk-your-openai-api-key-here
OPENAI_MODEL=gpt-4o-mini

# Langfuse 설정 (필수)
LANGFUSE_PUBLIC_KEY=pk-your-langfuse-public-key
LANGFUSE_SECRET_KEY=sk-your-langfuse-secret-key
LANGFUSE_HOST=https://your-langfuse-instance.com

# 사용자 식별
EMP_NO=EMP001

# LLM 설정
LLM_PROVIDER=OPENAI
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=2000

# Streamlit 설정
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=localhost

# A2A 서버 설정
A2A_ORCHESTRATOR_PORT=8100
A2A_PANDAS_ANALYST_PORT=8200
A2A_EDA_TOOLS_PORT=8203
A2A_DATA_VIZ_PORT=8202

# 로깅 설정
LOGGING_LEVEL=INFO
LOGGING_PROVIDER=langfuse
```

### 3. API 키 발급 가이드

#### OpenAI API 키
1. [OpenAI Platform](https://platform.openai.com/) 방문
2. 로그인 후 API Keys 섹션으로 이동
3. "Create new secret key" 클릭
4. 생성된 키를 .env 파일에 복사

#### Langfuse 설정
1. [Langfuse Cloud](https://cloud.langfuse.com/) 또는 Self-hosted 인스턴스 접속
2. 프로젝트 생성 또는 기존 프로젝트 선택
3. Settings → API Keys에서 Public/Secret 키 복사
4. .env 파일에 키와 호스트 URL 입력

### 4. 보안 설정

```bash
# .env 파일 권한 제한
chmod 600 .env

# Git에서 .env 파일 제외 확인
echo ".env" >> .gitignore
```

## 🎯 A2A 서버 시스템 설정

### 1. A2A 서버 시작

#### macOS/Linux
```bash
# A2A 서버 시스템 시작
./ai_ds_team_system_start.sh

# 백그라운드에서 시작
nohup ./ai_ds_team_system_start.sh > server.log 2>&1 &
```

#### Windows
```cmd
# PowerShell에서
.\ai_ds_team_system_start.bat

# 또는 개별 서버 시작
python a2a_orchestrator.py
```

### 2. 서버 상태 확인

```bash
# A2A 서버들 상태 확인
ps aux | grep python | grep -E "(8100|8200|8202|8203)"

# 또는 포트 사용 확인
netstat -ln | grep -E "(8100|8200|8202|8203)"

# 서버 응답 테스트
curl http://localhost:8100/.well-known/agent.json
curl http://localhost:8200/.well-known/agent.json
```

### 3. 로그 확인

```bash
# 서버 로그 확인
tail -f logs/orchestrator.log
tail -f logs/pandas_analyst.log
tail -f logs/eda_tools.log
```

## 🖥️ Streamlit UI 시작

### 1. UI 실행

```bash
# 기본 실행
streamlit run ai.py

# 커스텀 설정으로 실행
streamlit run ai.py --server.port 8501 --server.address 0.0.0.0

# 백그라운드 실행
nohup streamlit run ai.py > streamlit.log 2>&1 &
```

### 2. 웹 브라우저 접속

**기본 URL**: http://localhost:8501

**네트워크 접속 허용 시**: http://your-ip:8501

## ✅ 설치 검증

### 1. 자동 검증 스크립트

```bash
# 종합 호환성 테스트
python numpy_pandas_compatibility_test.py

# 실제 사용자 시나리오 테스트
python test_real_user_scenarios_simple.py

# A2A 통신 테스트
python test_a2a_communication.py
```

### 2. 수동 검증

#### 웹 UI 확인
1. http://localhost:8501 접속
2. 파일 업로드 기능 테스트
3. 샘플 분석 요청 수행

#### A2A 서버 확인
```bash
# 각 서버의 Agent Card 확인
curl -s http://localhost:8100/.well-known/agent.json | jq '.'
curl -s http://localhost:8200/.well-known/agent.json | jq '.'
curl -s http://localhost:8203/.well-known/agent.json | jq '.'
```

### 3. 성능 벤치마크

```bash
# 대용량 데이터 처리 테스트
python test_large_dataset_performance.py

# 동시 세션 테스트
python test_concurrent_sessions.py
```

## 🔧 문제 해결

### 일반적인 문제들

#### 1. 포트 충돌
```bash
# 포트 사용 확인
lsof -i :8501  # Streamlit
lsof -i :8100  # Orchestrator
lsof -i :8200  # Pandas Analyst

# 프로세스 종료
kill -9 <PID>

# 또는 시스템 종료 스크립트 사용
./ai_ds_team_system_stop.sh
```

#### 2. 의존성 충돌
```bash
# 가상환경 재생성
rm -rf .venv
uv venv
source .venv/bin/activate
uv pip install -e .
```

#### 3. API 키 문제
```bash
# 환경 변수 확인
echo $OPENAI_API_KEY
echo $LANGFUSE_PUBLIC_KEY

# .env 파일 확인
cat .env | grep -E "(OPENAI|LANGFUSE)"
```

#### 4. 권한 문제
```bash
# 필요한 디렉토리 권한 설정
chmod 755 ai_ds_team_system_start.sh
chmod 755 ai_ds_team_system_stop.sh
chmod 700 logs/
chmod 600 .env
```

### 성능 최적화

#### 1. 메모리 사용량 최적화
```bash
# Python 메모리 사용량 확인
python -c "
import psutil
print(f'사용 가능한 메모리: {psutil.virtual_memory().available / 1024**3:.2f}GB')
print(f'총 메모리: {psutil.virtual_memory().total / 1024**3:.2f}GB')
"
```

#### 2. CPU 최적화
```env
# .env에 추가
OMP_NUM_THREADS=4
NUMBA_NUM_THREADS=4
```

## 🔄 업데이트 및 유지보수

### 1. 시스템 업데이트

```bash
# Git 업데이트
git pull origin main

# 의존성 업데이트
uv pip install -e . --upgrade

# 서버 재시작
./ai_ds_team_system_stop.sh
./ai_ds_team_system_start.sh
```

### 2. 로그 관리

```bash
# 로그 로테이션
find logs/ -name "*.log" -mtime +7 -delete

# 로그 크기 확인
du -sh logs/
```

### 3. 백업

```bash
# 설정 백업
cp .env .env.backup
cp -r mcp-configs/ mcp-configs.backup/

# 데이터 백업
cp -r ai_ds_team/data/ data.backup/
```

## 🎉 설치 완료

설치가 성공적으로 완료되면 다음 상태가 됩니다:

✅ **Python 환경**: 3.12+ 가상환경 활성화  
✅ **의존성**: 모든 필수 패키지 설치  
✅ **환경 설정**: API 키 및 설정 완료  
✅ **A2A 서버**: 모든 에이전트 정상 작동  
✅ **Streamlit UI**: 웹 인터페이스 접근 가능  
✅ **테스트**: 모든 검증 테스트 통과  

### 다음 단계

1. [**사용자 가이드**](USER_GUIDE.md) - CherryAI 사용 방법 학습
2. [**API 문서**](API_REFERENCE.md) - 개발자를 위한 API 참조
3. [**문제 해결**](TROUBLESHOOTING.md) - 일반적인 문제 해결 방법

---

**🍒 CherryAI v2.0 설치 완료!** 이제 강력한 AI 기반 데이터 분석을 시작하세요. 