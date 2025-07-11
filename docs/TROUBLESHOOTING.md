# 🔧 CherryAI v2.0 문제 해결 가이드

**일반적인 문제들과 체계적인 해결 방법**

## 📋 목차

- [설치 관련 문제](#-설치-관련-문제)
- [A2A 서버 문제](#-a2a-서버-문제)
- [파일 업로드 문제](#-파일-업로드-문제)
- [성능 관련 문제](#-성능-관련-문제)
- [UI 및 브라우저 문제](#-ui-및-브라우저-문제)
- [API 및 연결 문제](#-api-및-연결-문제)
- [데이터 분석 문제](#-데이터-분석-문제)
- [로깅 및 추적 문제](#-로깅-및-추적-문제)
- [고급 문제 해결](#-고급-문제-해결)

## 🚀 설치 관련 문제

### 문제 1: Python 버전 호환성

**증상:**
```bash
ERROR: This package requires Python >=3.12
```

**해결책:**
```bash
# 1. Python 버전 확인
python --version

# 2. Python 3.12+ 설치 (macOS)
brew install python@3.12

# 3. Python 3.12+ 설치 (Ubuntu)
sudo apt update
sudo apt install python3.12 python3.12-venv

# 4. pyenv 사용 (권장)
pyenv install 3.12.10
pyenv local 3.12.10
```

### 문제 2: UV 패키지 매니저 설치 실패

**증상:**
```bash
uv: command not found
```

**해결책:**
```bash
# 방법 1: 공식 설치 스크립트
curl -LsSf https://astral.sh/uv/install.sh | sh

# 방법 2: pip로 설치
pip install uv

# 방법 3: 홈브루 (macOS)
brew install uv

# 설치 확인
uv --version
```

### 문제 3: 의존성 설치 실패

**증상:**
```bash
ERROR: Could not find a version that satisfies the requirement
```

**해결책:**
```bash
# 1. 가상환경 재생성
rm -rf .venv
uv venv
source .venv/bin/activate

# 2. 캐시 정리
uv cache clean

# 3. 개별 설치 시도
uv pip install pandas numpy streamlit

# 4. 호환성 확인된 버전 설치
uv pip install pandas==2.3.0 numpy==2.1.3 streamlit==1.46.0
```

### 문제 4: 권한 문제

**증상:**
```bash
Permission denied: '/usr/local/lib/python3.12'
```

**해결책:**
```bash
# 1. 가상환경 사용 (권장)
uv venv .venv
source .venv/bin/activate

# 2. 사용자 디렉토리에 설치
pip install --user package_name

# 3. sudo 사용 (비권장)
sudo pip install package_name
```

## 🤖 A2A 서버 문제

### 문제 1: A2A 서버 시작 실패

**증상:**
```bash
Error: Address already in use (port 8100)
```

**해결책:**
```bash
# 1. 포트 사용 확인
lsof -i :8100
netstat -ln | grep 8100

# 2. 프로세스 종료
kill -9 <PID>

# 3. 모든 CherryAI 프로세스 종료
./ai_ds_team_system_stop.sh

# 4. 포트 변경 (필요시)
export A2A_ORCHESTRATOR_PORT=8110
```

### 문제 2: Agent Card 접근 불가

**증상:**
```bash
curl: (7) Failed to connect to localhost port 8100
```

**진단:**
```bash
# 1. 서버 상태 확인
ps aux | grep python | grep 8100

# 2. 로그 확인
tail -f logs/orchestrator.log

# 3. 네트워크 연결 확인
curl -v http://localhost:8100/health
```

**해결책:**
```bash
# 1. 서버 재시작
python a2a_orchestrator.py

# 2. 방화벽 확인
sudo ufw status
sudo iptables -L

# 3. 환경변수 확인
echo $A2A_ORCHESTRATOR_PORT
```

### 문제 3: A2A 통신 오류

**증상:**
```json
{
  "error": "Agent not responding",
  "code": "AGENT_TIMEOUT"
}
```

**해결책:**
```bash
# 1. 모든 에이전트 상태 확인
curl http://localhost:8100/.well-known/agent.json
curl http://localhost:8200/.well-known/agent.json
curl http://localhost:8203/.well-known/agent.json

# 2. 타임아웃 설정 증가
export A2A_TIMEOUT=300

# 3. 개별 에이전트 재시작
python a2a_ds_servers/pandas_data_analyst_server.py
```

## 📁 파일 업로드 문제

### 문제 1: 파일 크기 제한

**증상:**
```
File size exceeds maximum limit (100MB)
```

**해결책:**
```python
# 1. Streamlit 설정 변경
# .streamlit/config.toml
[server]
maxUploadSize = 200

# 2. 파일 압축
gzip large_file.csv

# 3. 데이터 샘플링
df_sample = df.sample(n=10000)
df_sample.to_csv('sample_data.csv')
```

### 문제 2: 인코딩 문제

**증상:**
```
UnicodeDecodeError: 'utf-8' codec can't decode
```

**해결책:**
```python
# 1. 인코딩 감지 및 변환
import chardet

def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    return result['encoding']

# 2. 파일 재저장
import pandas as pd

# 다양한 인코딩 시도
encodings = ['utf-8', 'cp949', 'euc-kr', 'latin1']
for encoding in encodings:
    try:
        df = pd.read_csv(file_path, encoding=encoding)
        df.to_csv('fixed_file.csv', encoding='utf-8')
        break
    except UnicodeDecodeError:
        continue
```

### 문제 3: 파일 형식 인식 오류

**증상:**
```
File format not supported or corrupted
```

**해결책:**
```bash
# 1. 파일 타입 확인
file sample.csv
head -5 sample.csv

# 2. 구분자 확인
python -c "
import pandas as pd
try:
    df = pd.read_csv('sample.csv', nrows=5)
    print('기본 구분자 성공')
except:
    df = pd.read_csv('sample.csv', sep=';', nrows=5)
    print('세미콜론 구분자 성공')
"

# 3. Excel 파일 복구
python -c "
import pandas as pd
df = pd.read_excel('file.xlsx', engine='openpyxl')
df.to_csv('converted.csv')
"
```

## ⚡ 성능 관련 문제

### 문제 1: 느린 응답 속도

**증상:**
- 분석 요청 후 30초 이상 소요
- 브라우저가 응답 없음

**진단:**
```python
# 시스템 리소스 확인
import psutil

print(f"CPU 사용률: {psutil.cpu_percent()}%")
print(f"메모리 사용률: {psutil.virtual_memory().percent}%")
print(f"디스크 I/O: {psutil.disk_io_counters()}")
```

**해결책:**
```python
# 1. 데이터 샘플링
if len(df) > 50000:
    df_sample = df.sample(n=10000)
    print("대용량 데이터를 샘플링했습니다.")

# 2. 청크 단위 처리
def process_large_file(file_path, chunk_size=10000):
    chunks = []
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        processed_chunk = process_chunk(chunk)
        chunks.append(processed_chunk)
    return pd.concat(chunks)

# 3. 메모리 최적화
import gc
gc.collect()  # 가비지 컬렉션 강제 실행
```

### 문제 2: 메모리 부족

**증상:**
```
MemoryError: Unable to allocate array
```

**해결책:**
```python
# 1. 데이터 타입 최적화
def optimize_dtypes(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_numeric(df[col], downcast='integer')
            except:
                pass
        elif df[col].dtype == 'float64':
            df[col] = pd.to_numeric(df[col], downcast='float')
    return df

# 2. 청크 단위 처리
def memory_efficient_processing(file_path):
    result = None
    for chunk in pd.read_csv(file_path, chunksize=1000):
        processed = process_chunk(chunk)
        if result is None:
            result = processed
        else:
            result = pd.concat([result, processed])
        del chunk, processed
        gc.collect()
    return result

# 3. 스왑 메모리 확인 (Linux/Mac)
# sudo swapon --show
```

### 문제 3: A2A 에이전트 타임아웃

**증상:**
```
RequestTimeout: Agent did not respond within 30 seconds
```

**해결책:**
```python
# 1. 타임아웃 설정 증가
import os
os.environ['A2A_TIMEOUT'] = '300'  # 5분

# 2. 비동기 처리 개선
import asyncio

async def robust_agent_call(agent_url, request, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = await call_agent(agent_url, request, timeout=300)
            return response
        except asyncio.TimeoutError:
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(2 ** attempt)  # 지수 백오프

# 3. 에이전트 헬스체크
def check_agent_health():
    agents = ['8100', '8200', '8203', '8202']
    for port in agents:
        try:
            response = requests.get(f'http://localhost:{port}/health', timeout=5)
            print(f"Agent {port}: {response.status_code}")
        except:
            print(f"Agent {port}: OFFLINE")
```

## 🖥️ UI 및 브라우저 문제

### 문제 1: Streamlit 페이지 로드 실패

**증상:**
```
This site can't be reached
localhost refused to connect
```

**해결책:**
```bash
# 1. Streamlit 프로세스 확인
ps aux | grep streamlit

# 2. 포트 확인
lsof -i :8501

# 3. 수동 시작
streamlit run ai.py --server.port 8501 --server.address 0.0.0.0

# 4. 로그 확인
streamlit run ai.py --logger.level debug
```

### 문제 2: 세션 상태 초기화

**증상:**
- 업로드한 파일이 사라짐
- 분석 결과가 초기화됨

**해결책:**
```python
# 1. 세션 상태 디버깅
import streamlit as st

print("Current session state:")
for key, value in st.session_state.items():
    print(f"{key}: {type(value)}")

# 2. 세션 상태 유지
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = {}

if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}

# 3. 캐싱 사용
@st.cache_data
def load_and_cache_data(file_path):
    return pd.read_csv(file_path)
```

### 문제 3: 차트 렌더링 오류

**증상:**
```
PlotlyJSONEncoder: Object of type 'DataFrame' is not JSON serializable
```

**해결책:**
```python
# 1. 데이터 직렬화 확인
import json
import plotly.graph_objects as go

def safe_plotly_chart(fig):
    try:
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"차트 렌더링 오류: {e}")
        # 대안으로 matplotlib 사용
        st.pyplot(create_matplotlib_alternative(fig))

# 2. 데이터 타입 확인
def prepare_chart_data(df):
    # NaN 값 처리
    df = df.dropna()
    
    # 데이터 타입 변환
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str)
    
    return df
```

## 🔌 API 및 연결 문제

### 문제 1: OpenAI API 키 오류

**증상:**
```
AuthenticationError: Incorrect API key provided
```

**해결책:**
```bash
# 1. API 키 확인
echo $OPENAI_API_KEY

# 2. .env 파일 확인
cat .env | grep OPENAI_API_KEY

# 3. API 키 테스트
python -c "
import openai
import os
client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
response = client.chat.completions.create(
    model='gpt-3.5-turbo',
    messages=[{'role': 'user', 'content': 'test'}],
    max_tokens=10
)
print('API 키 정상 작동')
"

# 4. 새로운 API 키 발급
# https://platform.openai.com/api-keys 접속
```

### 문제 2: Langfuse 연결 실패

**증상:**
```
ConnectionError: Unable to connect to Langfuse host
```

**해결책:**
```python
# 1. 연결 테스트
import requests
import os

langfuse_host = os.getenv('LANGFUSE_HOST')
public_key = os.getenv('LANGFUSE_PUBLIC_KEY')

try:
    response = requests.get(f"{langfuse_host}/api/public/health")
    print(f"Langfuse 상태: {response.status_code}")
except Exception as e:
    print(f"Langfuse 연결 실패: {e}")

# 2. 환경변수 확인
required_vars = ['LANGFUSE_HOST', 'LANGFUSE_PUBLIC_KEY', 'LANGFUSE_SECRET_KEY']
for var in required_vars:
    value = os.getenv(var)
    print(f"{var}: {'설정됨' if value else '누락'}")

# 3. 대안 설정
# Self-hosted Langfuse 사용 시
os.environ['LANGFUSE_HOST'] = 'http://localhost:3000'
```

### 문제 3: 네트워크 프록시 문제

**증상:**
```
ProxyError: Cannot connect to proxy
```

**해결책:**
```bash
# 1. 프록시 설정 확인
echo $http_proxy
echo $https_proxy

# 2. 프록시 우회 설정
export no_proxy="localhost,127.0.0.1"

# 3. Python requests 프록시 설정
python -c "
import requests
import os

proxies = {
    'http': os.getenv('http_proxy'),
    'https': os.getenv('https_proxy')
}

response = requests.get('https://api.openai.com/v1/models', 
                       proxies=proxies, 
                       timeout=30)
print(f'연결 성공: {response.status_code}')
"
```

## 📊 데이터 분석 문제

### 문제 1: 분석 결과가 부정확함

**증상:**
- 통계 수치가 예상과 다름
- 차트가 잘못 표시됨

**진단:**
```python
# 1. 데이터 품질 확인
def diagnose_data_quality(df):
    print("=== 데이터 품질 진단 ===")
    print(f"총 행 수: {len(df)}")
    print(f"총 열 수: {len(df.columns)}")
    print(f"결측값: {df.isnull().sum().sum()}")
    print(f"중복 행: {df.duplicated().sum()}")
    print("\n데이터 타입:")
    print(df.dtypes)
    print("\n기본 통계:")
    print(df.describe())

# 2. 이상치 탐지
def detect_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower) | (df[column] > upper)]
    print(f"{column} 이상치: {len(outliers)}개")
    return outliers
```

### 문제 2: 메모리 오류로 분석 중단

**증상:**
```
MemoryError during statistical computation
```

**해결책:**
```python
# 1. 청크 단위 통계 계산
def chunk_statistics(df, chunk_size=10000):
    chunks = [df[i:i+chunk_size] for i in range(0, len(df), chunk_size)]
    
    # 평균 계산
    means = [chunk.mean() for chunk in chunks]
    overall_mean = pd.concat(means).mean()
    
    return overall_mean

# 2. 샘플링 기반 분석
def sample_analysis(df, sample_size=10000):
    if len(df) > sample_size:
        sample_df = df.sample(n=sample_size, random_state=42)
        print(f"샘플 크기: {len(sample_df)}")
        return sample_df
    return df

# 3. 스파스 매트릭스 사용
from scipy.sparse import csr_matrix

def optimize_categorical_data(df):
    for col in df.select_dtypes(include=['object']):
        df[col] = pd.Categorical(df[col])
    return df
```

### 문제 3: 시각화 생성 실패

**증상:**
```
PlotlyError: Invalid figure specification
```

**해결책:**
```python
# 1. 안전한 시각화 함수
def safe_visualization(df, chart_type='bar', x_col=None, y_col=None):
    try:
        if chart_type == 'bar' and x_col and y_col:
            fig = px.bar(df, x=x_col, y=y_col)
        elif chart_type == 'line' and x_col and y_col:
            fig = px.line(df, x=x_col, y=y_col)
        else:
            # 기본 차트
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) >= 2:
                fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1])
            else:
                fig = px.histogram(df, x=numeric_cols[0])
        
        return fig
    except Exception as e:
        st.error(f"시각화 생성 실패: {e}")
        return None

# 2. 데이터 전처리
def prepare_visualization_data(df):
    # 무한대 값 제거
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # NaN 값 처리
    df = df.dropna()
    
    # 너무 큰 값 처리
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].max() > 1e10:
            df[col] = df[col].clip(upper=df[col].quantile(0.99))
    
    return df
```

## 📝 로깅 및 추적 문제

### 문제 1: Langfuse 트레이스 누락

**증상:**
- 분석 과정이 Langfuse에 기록되지 않음
- 대시보드에 데이터가 없음

**해결책:**
```python
# 1. 수동 트레이스 확인
from core.enhanced_langfuse_tracer import get_enhanced_tracer

tracer = get_enhanced_tracer()

# 연결 테스트
try:
    test_trace = tracer.start_span("test_trace", TraceLevel.SYSTEM)
    tracer.end_span(test_trace, {"test": "success"})
    print("Langfuse 연결 정상")
except Exception as e:
    print(f"Langfuse 연결 실패: {e}")

# 2. 환경변수 재확인
import os
required_env = ['LANGFUSE_PUBLIC_KEY', 'LANGFUSE_SECRET_KEY', 'LANGFUSE_HOST']
for env_var in required_env:
    if not os.getenv(env_var):
        print(f"누락된 환경변수: {env_var}")

# 3. 대안 로깅
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fallback_logging(operation, data):
    logger.info(f"Operation: {operation}, Data: {data}")
```

### 문제 2: 로그 파일 접근 불가

**증상:**
```
PermissionError: [Errno 13] Permission denied: 'logs/system.log'
```

**해결책:**
```bash
# 1. 로그 디렉토리 권한 확인
ls -la logs/

# 2. 권한 수정
chmod 755 logs/
chmod 644 logs/*.log

# 3. 로그 디렉토리 재생성
rm -rf logs/
mkdir logs
chmod 755 logs/

# 4. 대안 로그 위치
export LOG_DIR=/tmp/cherryai_logs
mkdir -p $LOG_DIR
```

## 🔬 고급 문제 해결

### 시스템 진단 스크립트

```python
#!/usr/bin/env python3
"""
CherryAI 시스템 종합 진단 스크립트
"""

import subprocess
import requests
import psutil
import pandas as pd
import numpy as np
import sys
import os

def comprehensive_diagnosis():
    print("🔍 CherryAI 시스템 종합 진단")
    print("=" * 50)
    
    # 1. Python 환경 진단
    print(f"Python 버전: {sys.version}")
    print(f"Pandas 버전: {pd.__version__}")
    print(f"NumPy 버전: {np.__version__}")
    
    # 2. 시스템 리소스 진단
    print(f"\n💻 시스템 리소스:")
    print(f"CPU 사용률: {psutil.cpu_percent()}%")
    print(f"메모리 사용률: {psutil.virtual_memory().percent}%")
    print(f"디스크 사용률: {psutil.disk_usage('/').percent}%")
    
    # 3. A2A 서버 상태 진단
    print(f"\n🤖 A2A 서버 상태:")
    servers = {
        "Orchestrator": 8100,
        "Pandas Analyst": 8200,
        "EDA Tools": 8203,
        "Data Visualization": 8202
    }
    
    for name, port in servers.items():
        try:
            response = requests.get(f"http://localhost:{port}/health", timeout=5)
            status = "✅ 정상" if response.status_code == 200 else f"⚠️ 응답 코드: {response.status_code}"
        except:
            status = "❌ 접근 불가"
        print(f"{name} (포트 {port}): {status}")
    
    # 4. 환경변수 진단
    print(f"\n🔑 환경변수 상태:")
    required_env = [
        'OPENAI_API_KEY', 'LANGFUSE_PUBLIC_KEY', 
        'LANGFUSE_SECRET_KEY', 'LANGFUSE_HOST'
    ]
    
    for env_var in required_env:
        value = os.getenv(env_var)
        status = "✅ 설정됨" if value else "❌ 누락"
        print(f"{env_var}: {status}")
    
    # 5. 네트워크 진단
    print(f"\n🌐 네트워크 연결:")
    try:
        response = requests.get("https://api.openai.com/v1/models", timeout=10)
        print("OpenAI API: ✅ 연결 성공")
    except:
        print("OpenAI API: ❌ 연결 실패")
    
    # 6. 파일 시스템 진단
    print(f"\n📁 파일 시스템:")
    directories = ['logs/', 'ai_ds_team/data/', 'artifacts/']
    for directory in directories:
        if os.path.exists(directory):
            print(f"{directory}: ✅ 존재")
        else:
            print(f"{directory}: ❌ 누락")

if __name__ == "__main__":
    comprehensive_diagnosis()
```

### 자동 복구 스크립트

```bash
#!/bin/bash
# auto_recovery.sh - 자동 복구 스크립트

echo "🔧 CherryAI 자동 복구 시작"

# 1. 프로세스 정리
echo "1️⃣ 기존 프로세스 정리"
pkill -f streamlit
pkill -f "python.*a2a.*server"
sleep 3

# 2. 포트 정리
echo "2️⃣ 포트 정리"
ports=(8100 8200 8202 8203 8501)
for port in "${ports[@]}"; do
    pid=$(lsof -ti:$port)
    if [ ! -z "$pid" ]; then
        kill -9 $pid
        echo "포트 $port 정리 완료"
    fi
done

# 3. 캐시 정리
echo "3️⃣ 캐시 정리"
rm -rf __pycache__
rm -rf */__pycache__
rm -rf .streamlit/

# 4. 로그 디렉토리 재생성
echo "4️⃣ 로그 디렉토리 설정"
mkdir -p logs
chmod 755 logs

# 5. 가상환경 확인
echo "5️⃣ 가상환경 확인"
if [ ! -d ".venv" ]; then
    echo "가상환경 재생성"
    uv venv
fi

source .venv/bin/activate

# 6. 핵심 패키지 재설치
echo "6️⃣ 핵심 패키지 확인"
uv pip install --upgrade streamlit pandas numpy a2a-sdk

# 7. 서버 재시작
echo "7️⃣ 서버 시스템 재시작"
./ai_ds_team_system_start.sh

# 8. 건강성 체크
echo "8️⃣ 시스템 건강성 체크"
sleep 10
python -c "
import requests
import time

servers = [8100, 8200, 8203, 8202]
for port in servers:
    try:
        response = requests.get(f'http://localhost:{port}/health', timeout=5)
        print(f'포트 {port}: ✅ 정상')
    except:
        print(f'포트 {port}: ❌ 오류')
"

echo "🎉 자동 복구 완료"
```

### 성능 모니터링

```python
# performance_monitor.py - 실시간 성능 모니터링

import time
import psutil
import requests
from datetime import datetime

class PerformanceMonitor:
    def __init__(self):
        self.start_time = time.time()
        self.metrics = []
    
    def collect_metrics(self):
        """시스템 메트릭 수집"""
        current_time = datetime.now()
        
        # 시스템 메트릭
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # A2A 서버 응답 시간
        server_response_times = {}
        servers = [8100, 8200, 8203, 8202]
        
        for port in servers:
            try:
                start = time.time()
                requests.get(f'http://localhost:{port}/health', timeout=5)
                response_time = (time.time() - start) * 1000
                server_response_times[port] = response_time
            except:
                server_response_times[port] = None
        
        metrics = {
            'timestamp': current_time,
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'disk_percent': disk.percent,
            'server_response_times': server_response_times
        }
        
        self.metrics.append(metrics)
        return metrics
    
    def check_thresholds(self, metrics):
        """임계값 확인 및 알림"""
        alerts = []
        
        if metrics['cpu_percent'] > 80:
            alerts.append(f"⚠️ 높은 CPU 사용률: {metrics['cpu_percent']}%")
        
        if metrics['memory_percent'] > 85:
            alerts.append(f"⚠️ 높은 메모리 사용률: {metrics['memory_percent']}%")
        
        for port, response_time in metrics['server_response_times'].items():
            if response_time is None:
                alerts.append(f"❌ 서버 {port} 응답 없음")
            elif response_time > 5000:
                alerts.append(f"⚠️ 서버 {port} 느린 응답: {response_time:.1f}ms")
        
        return alerts
    
    def run_monitoring(self, duration_minutes=60):
        """지정된 시간 동안 모니터링 실행"""
        end_time = time.time() + (duration_minutes * 60)
        
        while time.time() < end_time:
            metrics = self.collect_metrics()
            alerts = self.check_thresholds(metrics)
            
            if alerts:
                print(f"\n🚨 {metrics['timestamp']} 알림:")
                for alert in alerts:
                    print(f"  {alert}")
            else:
                print(f"✅ {metrics['timestamp']} 정상")
            
            time.sleep(30)  # 30초마다 체크

if __name__ == "__main__":
    monitor = PerformanceMonitor()
    monitor.run_monitoring(duration_minutes=30)
```

## 📞 지원 및 도움

### 1. 문제 보고

버그나 문제 발견 시:

1. **로그 수집**
```bash
# 시스템 로그 수집
python diagnosis_script.py > system_diagnosis.txt

# 에러 로그 수집
tail -100 logs/*.log > error_logs.txt
```

2. **환경 정보 수집**
```bash
python -c "
import sys, platform
print(f'OS: {platform.system()} {platform.release()}')
print(f'Python: {sys.version}')
print(f'Architecture: {platform.machine()}')
"
```

3. **재현 단계 기록**
- 정확한 오류 메시지
- 수행한 작업 순서
- 사용한 데이터 파일 정보

### 2. 커뮤니티 지원

- **GitHub Issues**: 기술적 문제 및 버그 리포트
- **토론 포럼**: 사용법 질문 및 경험 공유
- **문서 Wiki**: 추가 팁 및 사용 사례

### 3. 전문 지원

프로덕션 환경에서 중요한 문제 발생 시:
- 우선순위 지원 채널 이용
- 상세한 진단 보고서 제공
- 원격 지원 세션 예약

---

**🍒 CherryAI v2.0** - *문제가 발생하면 체계적으로 해결하세요*

*이 가이드로 해결되지 않는 문제가 있다면 GitHub Issues에 신고해 주세요.* 