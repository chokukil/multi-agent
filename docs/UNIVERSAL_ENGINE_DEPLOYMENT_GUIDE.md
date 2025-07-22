# Universal Engine Deployment Guide

## 🚀 개요

이 가이드는 Universal Engine을 다양한 환경에서 배포하고 운영하는 방법을 제공합니다. 개발, 스테이징, 프로덕션 환경별 배포 전략과 모니터링 방법을 다룹니다.

## 📋 시스템 요구사항

### 최소 요구사항

#### 하드웨어
- **CPU**: 4코어 이상
- **메모리**: 8GB RAM
- **디스크**: 20GB SSD
- **네트워크**: 1Gbps

#### 소프트웨어
- **OS**: Ubuntu 20.04 LTS 이상, macOS 12+, Windows 10+
- **Python**: 3.9 이상
- **Docker**: 20.10 이상 (선택사항)
- **Git**: 2.30 이상

### 권장 요구사항

#### 하드웨어
- **CPU**: 8코어 이상
- **메모리**: 16GB RAM
- **디스크**: 100GB NVMe SSD
- **GPU**: NVIDIA RTX 3060 이상 (선택사항)
- **네트워크**: 10Gbps

#### 소프트웨어
- **OS**: Ubuntu 22.04 LTS
- **Python**: 3.11
- **Docker Compose**: 2.0 이상
- **Kubernetes**: 1.25+ (클러스터 배포 시)

## 🛠 배포 준비

### 1. 환경 설정

#### Python 가상환경 설정
```bash
# Python 3.11 설치 (Ubuntu)
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-dev

# 가상환경 생성
python3.11 -m venv universal_engine_env
source universal_engine_env/bin/activate

# pip 업그레이드
pip install --upgrade pip setuptools wheel
```

#### 시스템 패키지 설치
```bash
# Ubuntu/Debian
sudo apt install -y build-essential git curl wget

# macOS
brew install git curl wget

# CentOS/RHEL
sudo yum groupinstall -y "Development Tools"
sudo yum install -y git curl wget
```

### 2. 소스 코드 준비

```bash
# 저장소 클론
git clone <repository-url> universal-engine
cd universal-engine

# 의존성 설치
pip install -r requirements.txt

# 개발 의존성 설치 (개발 환경)
pip install -r requirements-dev.txt
```

### 3. 환경 변수 설정

#### `.env` 파일 생성
```bash
# 환경 설정 파일 생성
cp .env.example .env
```

#### 필수 환경 변수
```bash
# LLM 설정
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama2

# 또는 OpenAI 사용 시
# LLM_PROVIDER=openai
# OPENAI_API_KEY=your_openai_api_key

# A2A 에이전트 설정
A2A_PORT_START=8306
A2A_PORT_END=8315
A2A_TIMEOUT=30

# 데이터베이스 설정 (선택사항)
DATABASE_URL=sqlite:///./universal_engine.db

# 로깅 설정
LOG_LEVEL=INFO
LOG_FILE=/var/log/universal_engine.log

# 보안 설정
SECRET_KEY=your_secret_key_here
ENCRYPTION_KEY=your_encryption_key_here
```

## 🏗️ 배포 방법

### 1. 로컬 개발 배포

#### 기본 실행
```bash
# 가상환경 활성화
source universal_engine_env/bin/activate

# Ollama 서버 시작 (별도 터미널)
ollama serve

# Universal Engine 초기화 및 실행
python -m core.universal_engine.initialization.system_initializer
```

#### 개발 모드 실행
```bash
# 개발 모드로 실행 (핫 리로드)
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 2. Docker 배포

#### Dockerfile 생성
```dockerfile
FROM python:3.11-slim

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /app

# 의존성 파일 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 소스 코드 복사
COPY . .

# 환경 변수 설정
ENV PYTHONPATH=/app
ENV LOG_LEVEL=INFO

# 포트 노출
EXPOSE 8000 8306-8315

# 실행 명령
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### Docker Compose 설정
```yaml
version: '3.8'

services:
  universal-engine:
    build: .
    ports:
      - "8000:8000"
      - "8306-8315:8306-8315"
    environment:
      - LLM_PROVIDER=ollama
      - OLLAMA_BASE_URL=http://ollama:11434
    volumes:
      - ./logs:/app/logs
      - ./data:/app/data
    depends_on:
      - ollama
    restart: unless-stopped

  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    restart: unless-stopped
    command: ["ollama", "serve"]

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    restart: unless-stopped

volumes:
  ollama_data:
```

#### Docker 실행
```bash
# 빌드 및 실행
docker-compose up -d

# 로그 확인
docker-compose logs -f universal-engine

# 서비스 상태 확인
docker-compose ps
```

### 3. Kubernetes 배포

#### Namespace 생성
```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: universal-engine
```

#### ConfigMap 설정
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: universal-engine-config
  namespace: universal-engine
data:
  LLM_PROVIDER: "ollama"
  OLLAMA_BASE_URL: "http://ollama-service:11434"
  A2A_PORT_START: "8306"
  A2A_PORT_END: "8315"
  LOG_LEVEL: "INFO"
```

#### Deployment 설정
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: universal-engine
  namespace: universal-engine
spec:
  replicas: 3
  selector:
    matchLabels:
      app: universal-engine
  template:
    metadata:
      labels:
        app: universal-engine
    spec:
      containers:
      - name: universal-engine
        image: universal-engine:latest
        ports:
        - containerPort: 8000
        - containerPort: 8306
        - containerPort: 8315
        envFrom:
        - configMapRef:
            name: universal-engine-config
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

#### Service 설정
```yaml
apiVersion: v1
kind: Service
metadata:
  name: universal-engine-service
  namespace: universal-engine
spec:
  selector:
    app: universal-engine
  ports:
  - name: http
    port: 80
    targetPort: 8000
  - name: a2a-start
    port: 8306
    targetPort: 8306
  - name: a2a-end
    port: 8315
    targetPort: 8315
  type: LoadBalancer
```

#### Ingress 설정
```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: universal-engine-ingress
  namespace: universal-engine
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
spec:
  rules:
  - host: universal-engine.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: universal-engine-service
            port:
              number: 80
```

### 4. 클라우드 배포

#### AWS ECS 배포

**Task Definition**
```json
{
  "family": "universal-engine",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "2048",
  "memory": "4096",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "universal-engine",
      "image": "your-account.dkr.ecr.region.amazonaws.com/universal-engine:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "LLM_PROVIDER",
          "value": "openai"
        },
        {
          "name": "OPENAI_API_KEY",
          "valueFrom": "arn:aws:ssm:region:account:parameter/universal-engine/openai-api-key"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/universal-engine",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

#### Google Cloud Run 배포

```bash
# 이미지 빌드 및 푸시
docker build -t gcr.io/project-id/universal-engine .
docker push gcr.io/project-id/universal-engine

# Cloud Run 배포
gcloud run deploy universal-engine \
  --image gcr.io/project-id/universal-engine \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8000 \
  --memory 4Gi \
  --cpu 2 \
  --set-env-vars LLM_PROVIDER=openai,OPENAI_API_KEY=your-key
```

#### Azure Container Instances 배포

```bash
# 리소스 그룹 생성
az group create --name universal-engine-rg --location eastus

# 컨테이너 인스턴스 생성
az container create \
  --resource-group universal-engine-rg \
  --name universal-engine \
  --image universal-engine:latest \
  --cpu 2 \
  --memory 4 \
  --ports 8000 \
  --environment-variables LLM_PROVIDER=openai OPENAI_API_KEY=your-key
```

## 🔧 배포 후 설정

### 1. 헬스 체크 설정

#### 헬스 체크 엔드포인트
```python
@app.get("/health")
async def health_check():
    """시스템 헬스 체크"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "version": "1.0.0"
    }

@app.get("/ready")
async def readiness_check():
    """서비스 준비 상태 체크"""
    # LLM 서비스 연결 확인
    # A2A 에이전트 상태 확인
    # 데이터베이스 연결 확인
    return {"status": "ready"}
```

### 2. 로깅 설정

#### 로그 설정 파일 (`logging.conf`)
```ini
[loggers]
keys=root,universal_engine

[handlers]
keys=console,file,rotating_file

[formatters]
keys=detailed,simple

[logger_root]
level=INFO
handlers=console

[logger_universal_engine]
level=DEBUG
handlers=file,rotating_file
qualname=universal_engine
propagate=0

[handler_console]
class=StreamHandler
level=INFO
formatter=simple
args=(sys.stdout,)

[handler_file]
class=FileHandler
level=DEBUG
formatter=detailed
args=('/var/log/universal_engine.log',)

[handler_rotating_file]
class=handlers.RotatingFileHandler
level=DEBUG
formatter=detailed
args=('/var/log/universal_engine.log', 'a', 10485760, 5)

[formatter_detailed]
format=%(asctime)s %(name)-15s %(levelname)-8s %(processName)-10s %(message)s

[formatter_simple]
format=%(asctime)s - %(name)s - %(levelname)s - %(message)s
```

### 3. 보안 설정

#### SSL/TLS 설정
```bash
# Let's Encrypt 인증서 생성
sudo certbot --nginx -d universal-engine.yourdomain.com

# 또는 자체 서명 인증서 생성
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes
```

#### 방화벽 설정
```bash
# Ubuntu UFW 설정
sudo ufw allow 22/tcp
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw allow 8306:8315/tcp
sudo ufw enable

# 또는 iptables 설정
sudo iptables -A INPUT -p tcp --dport 8000 -j ACCEPT
sudo iptables -A INPUT -p tcp --dport 8306:8315 -j ACCEPT
```

## 📊 모니터링 및 관찰성

### 1. Prometheus 메트릭 수집

#### 메트릭 설정
```python
from prometheus_client import Counter, Histogram, Gauge, generate_latest

# 메트릭 정의
query_total = Counter('queries_total', 'Total queries processed')
query_duration = Histogram('query_duration_seconds', 'Query processing duration')
active_sessions = Gauge('active_sessions', 'Number of active sessions')

@app.get("/metrics")
async def metrics():
    """Prometheus 메트릭 엔드포인트"""
    return Response(generate_latest(), media_type="text/plain")
```

#### Prometheus 설정 (`prometheus.yml`)
```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'universal-engine'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
    scrape_interval: 10s
```

### 2. Grafana 대시보드

#### 대시보드 설정
```json
{
  "dashboard": {
    "title": "Universal Engine Dashboard",
    "panels": [
      {
        "title": "Query Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(queries_total[5m])",
            "legendFormat": "Queries/sec"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "query_duration_seconds",
            "legendFormat": "Response Time"
          }
        ]
      },
      {
        "title": "Active Sessions",
        "type": "singlestat",
        "targets": [
          {
            "expr": "active_sessions",
            "legendFormat": "Sessions"
          }
        ]
      }
    ]
  }
}
```

### 3. ELK Stack 로그 분석

#### Filebeat 설정 (`filebeat.yml`)
```yaml
filebeat.inputs:
- type: log
  enabled: true
  paths:
    - /var/log/universal_engine.log
  fields:
    service: universal-engine
  fields_under_root: true

output.elasticsearch:
  hosts: ["localhost:9200"]

processors:
  - add_host_metadata:
      when.not.contains.tags: forwarded
```

#### Logstash 파이프라인
```ruby
input {
  beats {
    port => 5044
  }
}

filter {
  if [service] == "universal-engine" {
    grok {
      match => { "message" => "%{TIMESTAMP_ISO8601:timestamp} %{WORD:logger} %{LOGLEVEL:level} %{GREEDYDATA:message}" }
    }
    
    date {
      match => [ "timestamp", "ISO8601" ]
    }
  }
}

output {
  elasticsearch {
    hosts => ["localhost:9200"]
    index => "universal-engine-%{+YYYY.MM.dd}"
  }
}
```

### 4. 알림 설정

#### AlertManager 설정
```yaml
route:
  group_by: ['alertname']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'web.hook'

receivers:
- name: 'web.hook'
  slack_configs:
  - api_url: 'https://hooks.slack.com/your-webhook-url'
    channel: '#universal-engine-alerts'
    title: 'Universal Engine Alert'
    text: '{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'
```

#### 알림 규칙
```yaml
groups:
- name: universal-engine
  rules:
  - alert: HighQueryLatency
    expr: query_duration_seconds > 10
    for: 2m
    annotations:
      summary: "High query latency detected"
      
  - alert: LowSuccessRate
    expr: rate(queries_total{status="success"}[5m]) / rate(queries_total[5m]) < 0.9
    for: 5m
    annotations:
      summary: "Success rate below 90%"
      
  - alert: ServiceDown
    expr: up{job="universal-engine"} == 0
    for: 1m
    annotations:
      summary: "Universal Engine service is down"
```

## 🔄 배포 자동화

### 1. CI/CD 파이프라인

#### GitHub Actions
```yaml
name: Deploy Universal Engine

on:
  push:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.11'
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
    - name: Run tests
      run: |
        python -m pytest tests/ -v --cov=core

  deploy:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
    - uses: actions/checkout@v2
    - name: Build Docker image
      run: docker build -t universal-engine:${{ github.sha }} .
    - name: Deploy to production
      run: |
        # 프로덕션 배포 스크립트 실행
        ./scripts/deploy.sh ${{ github.sha }}
```

#### Jenkins Pipeline
```groovy
pipeline {
    agent any
    
    stages {
        stage('Test') {
            steps {
                sh 'pip install -r requirements.txt'
                sh 'python -m pytest tests/'
            }
        }
        
        stage('Build') {
            steps {
                sh 'docker build -t universal-engine:${BUILD_NUMBER} .'
            }
        }
        
        stage('Deploy') {
            when {
                branch 'main'
            }
            steps {
                sh 'docker-compose -f docker-compose.prod.yml up -d'
            }
        }
    }
    
    post {
        failure {
            mail to: 'devops@company.com',
                 subject: "Build Failed: ${env.JOB_NAME} - ${env.BUILD_NUMBER}",
                 body: "Build failed. Check console output at ${env.BUILD_URL}"
        }
    }
}
```

### 2. Blue-Green 배포

#### 배포 스크립트
```bash
#!/bin/bash

# Blue-Green 배포 스크립트
set -e

VERSION=$1
CURRENT_ENV=$(curl -s http://localhost:8000/env | jq -r '.environment')

if [ "$CURRENT_ENV" = "blue" ]; then
    NEW_ENV="green"
    OLD_ENV="blue"
else
    NEW_ENV="blue"
    OLD_ENV="green"
fi

echo "Deploying to $NEW_ENV environment..."

# 새 환경에 배포
docker-compose -f docker-compose.$NEW_ENV.yml up -d
docker-compose -f docker-compose.$NEW_ENV.yml exec universal-engine python -c "
from core.universal_engine.initialization.system_initializer import UniversalEngineInitializer
import asyncio
asyncio.run(UniversalEngineInitializer().initialize_system())
"

# 헬스 체크
sleep 30
HEALTH=$(curl -s http://localhost:800$([[ "$NEW_ENV" = "blue" ]] && echo "1" || echo "2")/health | jq -r '.status')

if [ "$HEALTH" = "healthy" ]; then
    echo "Health check passed. Switching traffic..."
    # 로드 밸런서 설정 변경
    ./scripts/switch_traffic.sh $NEW_ENV
    
    # 이전 환경 정리
    sleep 60
    docker-compose -f docker-compose.$OLD_ENV.yml down
    
    echo "Deployment completed successfully!"
else
    echo "Health check failed. Rolling back..."
    docker-compose -f docker-compose.$NEW_ENV.yml down
    exit 1
fi
```

## 🔍 문제 해결

### 일반적인 배포 문제

#### 1. 포트 충돌
```bash
# 포트 사용 현황 확인
netstat -tulpn | grep 8306

# 포트를 사용하는 프로세스 종료
sudo kill -9 $(sudo lsof -t -i:8306)
```

#### 2. 메모리 부족
```bash
# 메모리 사용량 확인
free -h

# 스왑 생성 (임시 해결)
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

#### 3. LLM 서비스 연결 오류
```bash
# Ollama 서비스 상태 확인
systemctl status ollama

# Ollama 서비스 재시작
sudo systemctl restart ollama

# OpenAI API 키 확인
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
     https://api.openai.com/v1/models
```

### 성능 튜닝

#### 1. 시스템 최적화
```bash
# 파일 디스크립터 제한 증가
echo "* soft nofile 65536" >> /etc/security/limits.conf
echo "* hard nofile 65536" >> /etc/security/limits.conf

# 커널 파라미터 튜닝
echo "net.core.somaxconn = 65536" >> /etc/sysctl.conf
echo "net.ipv4.tcp_max_syn_backlog = 65536" >> /etc/sysctl.conf
sysctl -p
```

#### 2. Python 최적화
```bash
# PyPy 사용 (호환성 확인 필요)
pip install pypy3

# JIT 컴파일 활성화
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
```

## 📚 추가 리소스

### 관련 문서
- [Universal Engine User Guide](./UNIVERSAL_ENGINE_USER_GUIDE.md)
- [API Reference](./UNIVERSAL_ENGINE_API_REFERENCE.md)
- [Troubleshooting Guide](./UNIVERSAL_ENGINE_TROUBLESHOOTING.md)

### 모니터링 도구
- **Prometheus**: 메트릭 수집
- **Grafana**: 대시보드 및 시각화
- **ELK Stack**: 로그 분석
- **Jaeger**: 분산 추적

### 보안 도구
- **OWASP ZAP**: 보안 스캐닝
- **Bandit**: Python 코드 보안 분석
- **Safety**: 의존성 보안 체크

---

이 배포 가이드를 통해 Universal Engine을 안전하고 효율적으로 운영 환경에 배포할 수 있습니다. 추가 지원이 필요한 경우 기술 지원팀에 문의하시기 바랍니다.