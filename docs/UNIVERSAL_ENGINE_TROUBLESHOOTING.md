# Universal Engine Troubleshooting Guide

## 🚨 개요

이 가이드는 Universal Engine 사용 중 발생할 수 있는 일반적인 문제들과 해결 방법을 제공합니다. 문제 진단부터 해결까지 단계별로 설명합니다.

## 🔍 일반적인 문제 해결

### 1. 시스템 초기화 문제

#### 🔴 문제: System initialization failed
```
Error: Universal Engine initialization failed
```

**원인:**
- LLM 서비스 연결 실패
- 필수 환경 변수 누락
- 포트 충돌

**해결 방법:**
```bash
# 1. 환경 변수 확인
echo $LLM_PROVIDER
echo $OLLAMA_BASE_URL

# 2. Ollama 서비스 상태 확인
curl http://localhost:11434/api/version

# 3. 포트 충돌 확인
netstat -tulpn | grep -E ":(8306|8307|8308|8309|8310|8311|8312|8313|8314|8315)\s"

# 4. 시스템 재초기화
python -c "
from core.universal_engine.initialization.system_initializer import UniversalEngineInitializer
import asyncio
asyncio.run(UniversalEngineInitializer().initialize_system())
"
```

#### 🔴 문제: A2A agents not discovered
```
Error: No A2A agents found in port range 8306-8315
```

**해결 방법:**
```bash
# 1. 포트 범위 확인
for port in {8306..8315}; do
  nc -z localhost $port && echo "Port $port is open" || echo "Port $port is closed"
done

# 2. A2A 에이전트 수동 시작 (개발 모드)
python scripts/start_a2a_agents.py

# 3. 방화벽 확인 및 포트 허용
sudo ufw allow 8306:8315/tcp
```

### 2. LLM 연결 문제

#### 🔴 문제: LLM service unavailable
```
Error: Failed to connect to LLM service
```

**Ollama 사용 시:**
```bash
# 1. Ollama 서비스 상태 확인
systemctl status ollama

# 2. Ollama 서비스 시작
ollama serve

# 3. 모델 다운로드 확인
ollama list

# 4. 모델이 없는 경우 다운로드
ollama pull llama2

# 5. 연결 테스트
curl -X POST http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{"model": "llama2", "prompt": "Hello", "stream": false}'
```

**OpenAI 사용 시:**
```bash
# 1. API 키 확인
echo $OPENAI_API_KEY

# 2. API 연결 테스트
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
     https://api.openai.com/v1/models

# 3. 환경 변수 설정 확인
export LLM_PROVIDER=openai
export OPENAI_API_KEY=your_api_key_here
```

#### 🔴 문제: Token limit exceeded
```
Error: Request exceeds token limit
```

**해결 방법:**
```python
# 1. 입력 데이터 크기 확인 및 청크 분할
import pandas as pd

def chunk_dataframe(df, max_rows=1000):
    """대용량 데이터를 청크 단위로 분할"""
    for i in range(0, len(df), max_rows):
        yield df[i:i + max_rows]

# 2. 쿼리 길이 제한
def truncate_query(query, max_tokens=2000):
    """쿼리 길이 제한"""
    words = query.split()
    if len(words) > max_tokens:
        return ' '.join(words[:max_tokens]) + "..."
    return query

# 3. 컨텍스트 최적화
context = {
    "session_id": "sess123",
    "summarized_history": True,  # 긴 히스토리 요약
    "essential_only": True       # 필수 정보만 포함
}
```

### 3. A2A 통합 시스템 문제

#### 🔴 문제: A2A agent timeout
```
Error: Timeout waiting for agent response on port 8307
```

**해결 방법:**
```python
# 1. 에이전트 상태 확인
from core.universal_engine.a2a_integration.a2a_agent_discovery import A2AAgentDiscoverySystem
import asyncio

async def check_agents():
    discovery = A2AAgentDiscoverySystem()
    status = await discovery.check_agent_health()
    for agent_id, info in status.items():
        print(f"Agent {agent_id}: {info['status']} (Port: {info.get('port', 'N/A')})")

asyncio.run(check_agents())

# 2. 타임아웃 증가
context = {
    "a2a_timeout": 60,  # 기본 30초에서 60초로 증가
    "retry_count": 3    # 재시도 횟수 증가
}

# 3. 에이전트 재시작
./scripts/restart_a2a_agents.sh
```

#### 🔴 문제: Circuit breaker is open
```
Warning: Circuit breaker open for agent on port 8309
```

**해결 방법:**
```python
# 1. Circuit Breaker 상태 확인
from core.universal_engine.a2a_integration.a2a_error_handler import A2AErrorHandler

error_handler = A2AErrorHandler()
# Circuit Breaker 상태 리셋 (관리자 권한 필요)

# 2. 문제 에이전트 수동 복구
import requests

try:
    response = requests.get("http://localhost:8309/health", timeout=5)
    if response.status_code == 200:
        print("Agent is healthy, resetting circuit breaker")
except requests.exceptions.RequestException as e:
    print(f"Agent still unhealthy: {e}")

# 3. 대체 에이전트 사용
context = {
    "exclude_agents": ["statistical_analyzer"],  # 문제 에이전트 제외
    "fallback_mode": True
}
```

### 4. 성능 관련 문제

#### 🔴 문제: Slow query processing
```
Warning: Query processing took 45.2 seconds (threshold: 10.0s)
```

**해결 방법:**
```python
# 1. 쿼리 복잡도 분석
def analyze_query_complexity(query, data):
    complexity_score = 0
    
    # 데이터 크기
    if hasattr(data, 'shape'):
        rows, cols = data.shape
        complexity_score += rows * 0.001 + cols * 0.01
    
    # 쿼리 키워드
    complex_keywords = ['machine learning', 'clustering', 'regression', 'neural network']
    for keyword in complex_keywords:
        if keyword.lower() in query.lower():
            complexity_score += 5
    
    return complexity_score

# 2. 데이터 샘플링
def sample_large_data(df, max_rows=10000):
    if len(df) > max_rows:
        return df.sample(n=max_rows)
    return df

# 3. 캐싱 활용
import hashlib
import pickle
import os

def cache_result(query, data, result):
    cache_key = hashlib.md5(f"{query}{str(data.shape) if hasattr(data, 'shape') else ''}".encode()).hexdigest()
    cache_file = f"cache/{cache_key}.pkl"
    os.makedirs("cache", exist_ok=True)
    with open(cache_file, 'wb') as f:
        pickle.dump(result, f)

def get_cached_result(query, data):
    cache_key = hashlib.md5(f"{query}{str(data.shape) if hasattr(data, 'shape') else ''}".encode()).hexdigest()
    cache_file = f"cache/{cache_key}.pkl"
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    return None
```

#### 🔴 문제: High memory usage
```
Error: Memory usage exceeded 80% threshold (current: 85%)
```

**해결 방법:**
```bash
# 1. 메모리 사용량 모니터링
ps aux | grep python | sort -nrk 4 | head -5

# 2. 메모리 프로파일링
pip install memory-profiler
python -m memory_profiler your_script.py

# 3. 가비지 컬렉션 강제 실행
python -c "import gc; gc.collect()"

# 4. 시스템 스왑 확인 및 추가
free -h
sudo swapon --show

# 임시 스왑 파일 생성
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### 5. 데이터 관련 문제

#### 🔴 문제: Data format not supported
```
Error: Unsupported data format: .xlsx
```

**해결 방법:**
```python
# 1. 지원 형식 확인
supported_formats = ['.csv', '.json', '.parquet', '.pkl']

# 2. 데이터 형식 변환
import pandas as pd

# Excel 파일 읽기
if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
    df = pd.read_excel(file_path)
    # CSV로 변환하여 저장
    csv_path = file_path.replace('.xlsx', '.csv').replace('.xls', '.csv')
    df.to_csv(csv_path, index=False)
    
# JSON 파일 처리
elif file_path.endswith('.json'):
    df = pd.read_json(file_path)

# 3. 데이터 유효성 검사
def validate_data(df):
    issues = []
    
    # 빈 데이터프레임
    if df.empty:
        issues.append("DataFrame is empty")
    
    # 너무 많은 결측값
    missing_ratio = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
    if missing_ratio > 0.5:
        issues.append(f"High missing values ratio: {missing_ratio:.2%}")
    
    # 데이터 타입 문제
    object_cols = df.select_dtypes(include=['object']).columns
    if len(object_cols) == len(df.columns):
        issues.append("All columns are object type")
    
    return issues
```

#### 🔴 문제: Data corruption detected
```
Error: Data integrity check failed
```

**해결 방법:**
```python
# 1. 데이터 무결성 검사
import pandas as pd
import numpy as np

def data_integrity_check(df):
    report = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicate_rows': df.duplicated().sum(),
        'infinite_values': {}
    }
    
    # 무한값 검사
    for col in df.select_dtypes(include=[np.number]).columns:
        inf_count = np.isinf(df[col]).sum()
        if inf_count > 0:
            report['infinite_values'][col] = inf_count
    
    return report

# 2. 데이터 정리
def clean_data(df):
    # 중복 행 제거
    df = df.drop_duplicates()
    
    # 무한값 처리
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # 결측값 처리 (수치형은 중앙값, 범주형은 최빈값)
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown', inplace=True)
    
    return df
```

### 6. 보안 관련 문제

#### 🔴 문제: Malicious input detected
```
Warning: Potentially malicious input blocked
```

**해결 방법:**
```python
# 1. 입력 검증 로직 확인
import re

def is_malicious_input(query):
    malicious_patterns = [
        r"'; *DROP +TABLE",  # SQL Injection
        r"<script.*?>",      # XSS
        r"\.\.\/",           # Path Traversal
        r"system\s*\(",      # Command Injection
        r"\$\{.*\}",         # Template Injection
    ]
    
    for pattern in malicious_patterns:
        if re.search(pattern, query, re.IGNORECASE):
            return True, pattern
    return False, None

# 2. 입력 정화
def sanitize_input(query):
    # 위험한 문자 제거
    query = re.sub(r"[<>\"';\\]", "", query)
    # 길이 제한
    if len(query) > 10000:
        query = query[:10000] + "..."
    return query

# 3. 허용 목록 기반 필터링
allowed_keywords = [
    'analyze', 'show', 'display', 'calculate', 'summarize',
    'plot', 'chart', 'graph', 'trend', 'pattern', 'correlation'
]

def validate_query_keywords(query):
    words = query.lower().split()
    has_allowed = any(keyword in ' '.join(words) for keyword in allowed_keywords)
    return has_allowed
```

#### 🔴 문제: Session security violation
```
Error: Session validation failed
```

**해결 방법:**
```python
# 1. 세션 유효성 검사
from datetime import datetime, timedelta
import hashlib

def validate_session(session_data):
    current_time = datetime.now()
    
    # 세션 만료 확인
    if 'expires_at' in session_data:
        expires_at = datetime.fromisoformat(session_data['expires_at'])
        if current_time > expires_at:
            return False, "Session expired"
    
    # 세션 ID 형식 확인
    if not re.match(r'^[a-zA-Z0-9_-]{16,}$', session_data.get('session_id', '')):
        return False, "Invalid session ID format"
    
    # 사용자 인증 확인
    if 'user_id' not in session_data or not session_data['user_id']:
        return False, "Missing user authentication"
    
    return True, "Valid session"

# 2. 세션 재생성
def regenerate_session(old_session):
    new_session = {
        'session_id': hashlib.sha256(f"{datetime.now()}{old_session['user_id']}".encode()).hexdigest()[:32],
        'user_id': old_session['user_id'],
        'created_at': datetime.now().isoformat(),
        'expires_at': (datetime.now() + timedelta(hours=24)).isoformat(),
        'messages': [],
        'user_profile': old_session.get('user_profile', {})
    }
    return new_session
```

## 🛠 진단 도구

### 1. 시스템 상태 확인 스크립트

```python
#!/usr/bin/env python3
"""
Universal Engine 시스템 진단 도구
"""

import asyncio
import requests
import sys
import json
from datetime import datetime
import pandas as pd

async def system_diagnostics():
    """시스템 전반적인 상태 진단"""
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'system_status': {},
        'component_status': {},
        'recommendations': []
    }
    
    print("🔍 Universal Engine 시스템 진단 시작...")
    
    # 1. LLM 서비스 확인
    print("\n1. LLM 서비스 상태 확인...")
    try:
        response = requests.get("http://localhost:11434/api/version", timeout=5)
        if response.status_code == 200:
            report['system_status']['llm_service'] = 'healthy'
            print("   ✅ LLM 서비스 정상")
        else:
            report['system_status']['llm_service'] = 'unhealthy'
            print("   ❌ LLM 서비스 응답 오류")
    except requests.exceptions.RequestException:
        report['system_status']['llm_service'] = 'unreachable'
        print("   ❌ LLM 서비스 연결 실패")
        report['recommendations'].append("Ollama 서비스를 확인하고 재시작하세요")
    
    # 2. A2A 에이전트 상태 확인
    print("\n2. A2A 에이전트 상태 확인...")
    agent_ports = range(8306, 8316)
    healthy_agents = 0
    
    for port in agent_ports:
        try:
            response = requests.get(f"http://localhost:{port}/health", timeout=2)
            if response.status_code == 200:
                healthy_agents += 1
                print(f"   ✅ Port {port}: 정상")
            else:
                print(f"   ❌ Port {port}: 응답 오류")
        except requests.exceptions.RequestException:
            print(f"   ❌ Port {port}: 연결 실패")
    
    report['system_status']['a2a_agents'] = f"{healthy_agents}/{len(agent_ports)}"
    if healthy_agents < len(agent_ports) * 0.8:
        report['recommendations'].append("A2A 에이전트 서비스를 확인하세요")
    
    # 3. 메모리 사용량 확인
    print("\n3. 시스템 리소스 확인...")
    import psutil
    
    memory_percent = psutil.virtual_memory().percent
    cpu_percent = psutil.cpu_percent(interval=1)
    disk_percent = psutil.disk_usage('/').percent
    
    report['system_status']['memory_usage'] = f"{memory_percent}%"
    report['system_status']['cpu_usage'] = f"{cpu_percent}%"
    report['system_status']['disk_usage'] = f"{disk_percent}%"
    
    print(f"   메모리 사용량: {memory_percent}%")
    print(f"   CPU 사용량: {cpu_percent}%")
    print(f"   디스크 사용량: {disk_percent}%")
    
    if memory_percent > 80:
        report['recommendations'].append("메모리 사용량이 높습니다. 시스템 최적화가 필요합니다")
    if disk_percent > 90:
        report['recommendations'].append("디스크 공간이 부족합니다")
    
    # 4. 컴포넌트 테스트
    print("\n4. Universal Engine 컴포넌트 테스트...")
    try:
        from core.universal_engine.universal_query_processor import UniversalQueryProcessor
        processor = UniversalQueryProcessor()
        
        test_data = pd.DataFrame({'test': [1, 2, 3]})
        result = await processor.process_query(
            query="테스트 쿼리",
            data=test_data,
            context={'diagnostic_test': True}
        )
        
        report['component_status']['query_processor'] = 'working'
        print("   ✅ Query Processor 정상")
        
    except Exception as e:
        report['component_status']['query_processor'] = f'error: {str(e)}'
        print(f"   ❌ Query Processor 오류: {e}")
        report['recommendations'].append("Universal Engine 컴포넌트를 재초기화하세요")
    
    # 진단 보고서 출력
    print("\n" + "="*60)
    print("📊 진단 결과 요약")
    print("="*60)
    
    for component, status in report['system_status'].items():
        print(f"{component}: {status}")
    
    if report['recommendations']:
        print("\n🔧 권장 조치사항:")
        for i, recommendation in enumerate(report['recommendations'], 1):
            print(f"{i}. {recommendation}")
    else:
        print("\n✅ 모든 시스템이 정상 상태입니다!")
    
    # 진단 결과 JSON 파일로 저장
    with open('system_diagnosis.json', 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n📄 상세 진단 결과가 'system_diagnosis.json'에 저장되었습니다")
    
    return report

if __name__ == "__main__":
    asyncio.run(system_diagnostics())
```

### 2. 로그 분석 스크립트

```python
#!/usr/bin/env python3
"""
Universal Engine 로그 분석 도구
"""

import re
import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta

def analyze_logs(log_file_path="/var/log/universal_engine.log"):
    """로그 파일 분석"""
    
    analysis_result = {
        'total_entries': 0,
        'error_count': 0,
        'warning_count': 0,
        'performance_issues': [],
        'frequent_errors': {},
        'recommendations': []
    }
    
    error_patterns = []
    response_times = []
    
    try:
        with open(log_file_path, 'r') as f:
            for line in f:
                analysis_result['total_entries'] += 1
                
                # 에러 패턴 추출
                if 'ERROR' in line:
                    analysis_result['error_count'] += 1
                    error_msg = re.search(r'ERROR.*?:(.*)', line)
                    if error_msg:
                        error_patterns.append(error_msg.group(1).strip())
                
                # 경고 패턴 추출
                elif 'WARNING' in line:
                    analysis_result['warning_count'] += 1
                
                # 응답 시간 추출
                response_time_match = re.search(r'processing took ([\d.]+) seconds', line)
                if response_time_match:
                    response_times.append(float(response_time_match.group(1)))
        
        # 자주 발생하는 에러 분석
        error_counter = Counter(error_patterns)
        analysis_result['frequent_errors'] = dict(error_counter.most_common(5))
        
        # 성능 이슈 분석
        if response_times:
            avg_response_time = sum(response_times) / len(response_times)
            slow_queries = [t for t in response_times if t > 10]
            
            analysis_result['average_response_time'] = avg_response_time
            analysis_result['slow_query_count'] = len(slow_queries)
            
            if avg_response_time > 5:
                analysis_result['recommendations'].append(
                    f"평균 응답 시간이 {avg_response_time:.2f}초로 높습니다. 성능 최적화가 필요합니다"
                )
        
        # 에러율 분석
        error_rate = analysis_result['error_count'] / analysis_result['total_entries'] * 100
        if error_rate > 5:
            analysis_result['recommendations'].append(
                f"에러율이 {error_rate:.1f}%로 높습니다. 시스템 안정성 검토가 필요합니다"
            )
        
    except FileNotFoundError:
        analysis_result['error'] = f"로그 파일을 찾을 수 없습니다: {log_file_path}"
    
    return analysis_result

def print_log_analysis(analysis):
    """로그 분석 결과 출력"""
    print("📊 로그 분석 결과")
    print("="*50)
    
    if 'error' in analysis:
        print(f"❌ {analysis['error']}")
        return
    
    print(f"총 로그 엔트리: {analysis['total_entries']:,}")
    print(f"에러 발생 수: {analysis['error_count']:,}")
    print(f"경고 발생 수: {analysis['warning_count']:,}")
    
    if 'average_response_time' in analysis:
        print(f"평균 응답 시간: {analysis['average_response_time']:.2f}초")
        print(f"느린 쿼리 수: {analysis['slow_query_count']:,}")
    
    if analysis['frequent_errors']:
        print("\n🔴 자주 발생하는 에러:")
        for error, count in analysis['frequent_errors'].items():
            print(f"  • {error} ({count}회)")
    
    if analysis['recommendations']:
        print("\n🔧 권장 조치사항:")
        for rec in analysis['recommendations']:
            print(f"  • {rec}")

if __name__ == "__main__":
    import sys
    log_file = sys.argv[1] if len(sys.argv) > 1 else "/var/log/universal_engine.log"
    analysis = analyze_logs(log_file)
    print_log_analysis(analysis)
```

### 3. 성능 모니터링 대시보드

```python
#!/usr/bin/env python3
"""
Universal Engine 실시간 성능 모니터링 대시보드
"""

import time
import psutil
import requests
from datetime import datetime
import json

class PerformanceMonitor:
    def __init__(self):
        self.metrics_history = []
    
    def collect_metrics(self):
        """시스템 메트릭 수집"""
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'system': {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_percent': psutil.disk_usage('/').percent,
                'load_average': psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0
            },
            'network': {
                'connections': len(psutil.net_connections()),
                'bytes_sent': psutil.net_io_counters().bytes_sent,
                'bytes_recv': psutil.net_io_counters().bytes_recv
            },
            'application': {
                'llm_status': self.check_llm_status(),
                'a2a_agents': self.check_a2a_agents(),
                'response_time': self.measure_response_time()
            }
        }
        
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > 100:  # 최근 100개만 유지
            self.metrics_history.pop(0)
        
        return metrics
    
    def check_llm_status(self):
        """LLM 서비스 상태 확인"""
        try:
            response = requests.get("http://localhost:11434/api/version", timeout=2)
            return "healthy" if response.status_code == 200 else "unhealthy"
        except:
            return "unreachable"
    
    def check_a2a_agents(self):
        """A2A 에이전트 상태 확인"""
        healthy_count = 0
        total_count = 10  # 8306-8315
        
        for port in range(8306, 8316):
            try:
                response = requests.get(f"http://localhost:{port}/health", timeout=1)
                if response.status_code == 200:
                    healthy_count += 1
            except:
                pass
        
        return f"{healthy_count}/{total_count}"
    
    def measure_response_time(self):
        """응답 시간 측정"""
        try:
            start_time = time.time()
            response = requests.get("http://localhost:8000/health", timeout=5)
            end_time = time.time()
            
            if response.status_code == 200:
                return round((end_time - start_time) * 1000, 2)  # ms 단위
        except:
            pass
        return None
    
    def display_dashboard(self, metrics):
        """실시간 대시보드 출력"""
        import os
        os.system('clear' if os.name == 'posix' else 'cls')
        
        print("🖥️  Universal Engine 실시간 모니터링 대시보드")
        print("=" * 60)
        print(f"📅 업데이트 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # 시스템 메트릭
        sys_metrics = metrics['system']
        print("🖥️  시스템 리소스:")
        print(f"   CPU 사용량:    {sys_metrics['cpu_percent']:6.1f}% {'🔴' if sys_metrics['cpu_percent'] > 80 else '🟢' if sys_metrics['cpu_percent'] < 50 else '🟡'}")
        print(f"   메모리 사용량: {sys_metrics['memory_percent']:6.1f}% {'🔴' if sys_metrics['memory_percent'] > 80 else '🟢' if sys_metrics['memory_percent'] < 70 else '🟡'}")
        print(f"   디스크 사용량: {sys_metrics['disk_percent']:6.1f}% {'🔴' if sys_metrics['disk_percent'] > 90 else '🟢' if sys_metrics['disk_percent'] < 70 else '🟡'}")
        
        # 애플리케이션 메트릭
        app_metrics = metrics['application']
        print("\n🚀 애플리케이션 상태:")
        
        llm_status = app_metrics['llm_status']
        llm_icon = '🟢' if llm_status == 'healthy' else '🔴'
        print(f"   LLM 서비스:    {llm_status:>12} {llm_icon}")
        
        a2a_status = app_metrics['a2a_agents']
        a2a_healthy = int(a2a_status.split('/')[0])
        a2a_total = int(a2a_status.split('/')[1])
        a2a_icon = '🟢' if a2a_healthy == a2a_total else '🟡' if a2a_healthy > a2a_total * 0.7 else '🔴'
        print(f"   A2A 에이전트:  {a2a_status:>12} {a2a_icon}")
        
        response_time = app_metrics['response_time']
        if response_time:
            rt_icon = '🟢' if response_time < 100 else '🟡' if response_time < 500 else '🔴'
            print(f"   응답 시간:     {response_time:>9.1f}ms {rt_icon}")
        else:
            print(f"   응답 시간:     {'N/A':>12} ⚪")
        
        # 알림 및 권장사항
        alerts = []
        if sys_metrics['cpu_percent'] > 80:
            alerts.append("CPU 사용량이 높습니다")
        if sys_metrics['memory_percent'] > 80:
            alerts.append("메모리 사용량이 높습니다")
        if llm_status != 'healthy':
            alerts.append("LLM 서비스에 문제가 있습니다")
        if response_time and response_time > 1000:
            alerts.append("응답 시간이 느립니다")
        
        if alerts:
            print("\n⚠️  알림:")
            for alert in alerts:
                print(f"   • {alert}")
        
        print(f"\n💡 Press Ctrl+C to stop monitoring")
        print("-" * 60)
    
    def run_monitor(self, interval=5):
        """모니터링 실행"""
        try:
            while True:
                metrics = self.collect_metrics()
                self.display_dashboard(metrics)
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\n\n모니터링을 중단했습니다.")
            
            # 메트릭 히스토리 저장
            with open(f'metrics_history_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json', 'w') as f:
                json.dump(self.metrics_history, f, indent=2, ensure_ascii=False)
            print("메트릭 히스토리가 저장되었습니다.")

if __name__ == "__main__":
    monitor = PerformanceMonitor()
    monitor.run_monitor()
```

## 📞 지원 요청

문제가 해결되지 않는 경우 다음 정보를 포함하여 지원을 요청하세요:

### 📋 필수 정보

1. **시스템 환경**
   - OS 및 버전
   - Python 버전
   - Universal Engine 버전

2. **오류 정보**
   - 정확한 오류 메시지
   - 발생 시각
   - 재현 단계

3. **로그 파일**
   ```bash
   # 최근 로그 추출
   tail -100 /var/log/universal_engine.log > error_logs.txt
   ```

4. **시스템 상태**
   ```bash
   # 진단 스크립트 실행
   python diagnostics.py > system_status.txt
   ```

### 📧 지원 채널

- **GitHub Issues**: 버그 리포트 및 기능 요청
- **기술 문서**: [Universal Engine Documentation](./README.md)
- **커뮤니티 포럼**: 사용자 간 정보 공유

---

이 문제 해결 가이드를 통해 대부분의 일반적인 문제들을 해결할 수 있습니다. 추가적인 도움이 필요하시면 언제든지 지원팀에 문의하시기 바랍니다.