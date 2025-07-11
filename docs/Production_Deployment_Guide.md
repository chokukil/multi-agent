# 🚀 CherryAI Phase 1-4 프로덕션 배포 가이드

## 📋 배포 전 체크리스트

### ✅ **시스템 요구사항**
- [ ] Python 3.12+ 설치
- [ ] uv 패키지 매니저 설치
- [ ] 최소 8GB RAM (권장 16GB+)
- [ ] 최소 4CPU 코어 (권장 8+)
- [ ] 50GB+ 디스크 공간

### ✅ **환경 변수 설정**
```bash
# .env 파일 생성
cp .env.example .env

# 필수 환경 변수
OPENAI_API_KEY=your_openai_api_key
LANGFUSE_SECRET_KEY=your_langfuse_secret_key
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key
LANGFUSE_HOST=https://your-langfuse-instance.com

# 선택적 환경 변수
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
A2A_ORCHESTRATOR_PORT=8100
A2A_BASE_PORTS_START=8306
```

### ✅ **보안 설정**
```bash
# .env 파일 권한 제한
chmod 600 .env

# 로그 디렉토리 권한 설정
chmod 755 logs/
chmod 644 logs/*.log

# 데이터 디렉토리 보안
chmod 700 ai_ds_team/data/
chmod 700 a2a_ds_servers/artifacts/
```

---

## 🏗️ 배포 아키텍처

### **권장 프로덕션 아키텍처**
```
┌─────────────────────────────────────────┐
│             Load Balancer               │
│            (Nginx/HAProxy)              │
└─────────────┬───────────────────────────┘
              │
┌─────────────▼───────────────────────────┐
│         Streamlit UI Server            │
│        (Port 8501, Multi-Instance)     │
└─────────────┬───────────────────────────┘
              │
┌─────────────▼───────────────────────────┐
│         A2A Orchestrator               │
│            (Port 8100)                 │
└─────────────┬───────────────────────────┘
              │
┌─────────────▼───────────────────────────┐
│      A2A Agent Servers Cluster         │
│  (Ports 8306-8314, Auto-Scaling)       │
└─────────────────────────────────────────┘
```

---

## 🐳 Docker 배포 (권장)

### **1. Dockerfile 생성**
```dockerfile
FROM python:3.12-slim

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# uv 설치
RUN pip install uv

# 작업 디렉토리 설정
WORKDIR /app

# 의존성 파일 복사
COPY requirements.txt .
COPY pyproject.toml .

# 의존성 설치
RUN uv pip install --system -r requirements.txt

# 애플리케이션 코드 복사
COPY . .

# 포트 노출
EXPOSE 8501 8100 8306-8314

# 헬스체크
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# 시작 명령
CMD ["./docker-entrypoint.sh"]
```

### **2. docker-compose.yml**
```yaml
version: '3.8'

services:
  cherryai:
    build: .
    ports:
      - "8501:8501"
      - "8100:8100"
      - "8306-8314:8306-8314"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - LANGFUSE_SECRET_KEY=${LANGFUSE_SECRET_KEY}
      - LANGFUSE_PUBLIC_KEY=${LANGFUSE_PUBLIC_KEY}
      - LANGFUSE_HOST=${LANGFUSE_HOST}
    volumes:
      - ./data:/app/ai_ds_team/data
      - ./logs:/app/logs
      - ./artifacts:/app/a2a_ds_servers/artifacts
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - cherryai
    restart: unless-stopped
```

### **3. docker-entrypoint.sh**
```bash
#!/bin/bash
set -e

# A2A 서버들 시작
echo "Starting A2A servers..."
./ai_ds_team_system_start.sh &

# 서버 시작 대기
sleep 15

# Streamlit UI 시작
echo "Starting Streamlit UI..."
streamlit run ai.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection false
```

---

## ⚙️ 시스템 서비스 배포

### **1. systemd 서비스 파일 생성**

#### **A2A 서버 서비스**
```ini
# /etc/systemd/system/cherryai-a2a.service
[Unit]
Description=CherryAI A2A Servers
After=network.target

[Service]
Type=forking
User=cherryai
Group=cherryai
WorkingDirectory=/opt/cherryai
ExecStart=/opt/cherryai/ai_ds_team_system_start.sh
ExecStop=/opt/cherryai/ai_ds_team_system_stop.sh
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

#### **Streamlit UI 서비스**
```ini
# /etc/systemd/system/cherryai-ui.service
[Unit]
Description=CherryAI Streamlit UI
After=network.target cherryai-a2a.service
Requires=cherryai-a2a.service

[Service]
Type=simple
User=cherryai
Group=cherryai
WorkingDirectory=/opt/cherryai
ExecStart=/opt/cherryai/.venv/bin/streamlit run ai.py --server.port 8501 --server.address 0.0.0.0
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### **2. 서비스 활성화**
```bash
# 서비스 등록
sudo systemctl daemon-reload
sudo systemctl enable cherryai-a2a.service
sudo systemctl enable cherryai-ui.service

# 서비스 시작
sudo systemctl start cherryai-a2a.service
sudo systemctl start cherryai-ui.service

# 상태 확인
sudo systemctl status cherryai-a2a.service
sudo systemctl status cherryai-ui.service
```

---

## 🔧 Nginx 설정

### **nginx.conf**
```nginx
events {
    worker_connections 1024;
}

http {
    upstream cherryai_ui {
        server 127.0.0.1:8501;
    }

    upstream cherryai_a2a {
        server 127.0.0.1:8100;
    }

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=ui_limit:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=api_limit:10m rate=30r/s;

    server {
        listen 80;
        server_name your-domain.com;
        
        # Redirect to HTTPS
        return 301 https://$server_name$request_uri;
    }

    server {
        listen 443 ssl http2;
        server_name your-domain.com;

        # SSL Configuration
        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;

        # Security Headers
        add_header X-Frame-Options DENY;
        add_header X-Content-Type-Options nosniff;
        add_header X-XSS-Protection "1; mode=block";
        add_header Strict-Transport-Security "max-age=31536000; includeSubDomains";

        # Main UI
        location / {
            limit_req zone=ui_limit burst=20 nodelay;
            proxy_pass http://cherryai_ui;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # WebSocket support
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_read_timeout 86400;
        }

        # A2A API
        location /a2a/ {
            limit_req zone=api_limit burst=50 nodelay;
            proxy_pass http://cherryai_a2a/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        # Health check
        location /health {
            access_log off;
            return 200 "healthy\n";
            add_header Content-Type text/plain;
        }
    }
}
```

---

## 📊 모니터링 설정

### **1. 시스템 모니터링**
```bash
# Prometheus + Grafana 설정
docker run -d \
  --name prometheus \
  -p 9090:9090 \
  -v ./prometheus.yml:/etc/prometheus/prometheus.yml \
  prom/prometheus

docker run -d \
  --name grafana \
  -p 3000:3000 \
  -e "GF_SECURITY_ADMIN_PASSWORD=admin" \
  grafana/grafana
```

### **2. 로그 모니터링**
```bash
# 로그 로테이션 설정
cat > /etc/logrotate.d/cherryai << EOF
/opt/cherryai/logs/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 cherryai cherryai
    postrotate
        systemctl reload cherryai-ui.service
    endscript
}
EOF
```

### **3. 알림 설정**
```python
# monitoring/alerts.py
import psutil
import requests
import time

def check_system_health():
    """시스템 상태 체크 및 알림"""
    cpu_usage = psutil.cpu_percent(interval=1)
    memory_usage = psutil.virtual_memory().percent
    disk_usage = psutil.disk_usage('/').percent
    
    if cpu_usage > 80:
        send_alert(f"High CPU usage: {cpu_usage}%")
    if memory_usage > 85:
        send_alert(f"High memory usage: {memory_usage}%")
    if disk_usage > 90:
        send_alert(f"High disk usage: {disk_usage}%")

def send_alert(message):
    """Slack/Discord 알림 전송"""
    webhook_url = "YOUR_WEBHOOK_URL"
    payload = {"text": f"🚨 CherryAI Alert: {message}"}
    requests.post(webhook_url, json=payload)
```

---

## 🔐 보안 강화

### **1. 방화벽 설정**
```bash
# UFW 방화벽 설정
sudo ufw enable
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS
sudo ufw deny 8501/tcp   # Streamlit 직접 접근 차단
sudo ufw deny 8100:8314/tcp  # A2A 서버 직접 접근 차단
```

### **2. 접근 제어**
```python
# security/auth.py
import streamlit as st
import hashlib
import secrets

def authenticate_user():
    """사용자 인증"""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    if not st.session_state.authenticated:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        if st.button("Login"):
            if verify_credentials(username, password):
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Invalid credentials")
        return False
    
    return True

def verify_credentials(username, password):
    """자격 증명 확인"""
    # 실제 구현에서는 안전한 방법으로 자격 증명 저장
    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    # ... 검증 로직
    return True
```

### **3. API 키 관리**
```bash
# AWS Secrets Manager 사용 예시
aws secretsmanager create-secret \
    --name "cherryai/production/api-keys" \
    --secret-string '{
        "openai_api_key": "your_openai_key",
        "langfuse_secret_key": "your_langfuse_secret",
        "langfuse_public_key": "your_langfuse_public"
    }'
```

---

## 🚀 배포 스크립트

### **deploy.sh**
```bash
#!/bin/bash
set -e

echo "🚀 CherryAI 프로덕션 배포 시작"

# 환경 변수 확인
if [[ -z "$OPENAI_API_KEY" ]]; then
    echo "❌ OPENAI_API_KEY가 설정되지 않았습니다"
    exit 1
fi

# 의존성 설치
echo "📦 의존성 설치 중..."
uv pip install -r requirements.txt

# 테스트 실행
echo "🧪 테스트 실행 중..."
python quick_integration_test.py
if [ $? -ne 0 ]; then
    echo "❌ 테스트 실패"
    exit 1
fi

# A2A 서버 재시작
echo "🔄 A2A 서버 재시작..."
./ai_ds_team_system_stop.sh
sleep 5
./ai_ds_team_system_start.sh

# 서버 상태 확인
echo "⏱️ 서버 시작 대기 중..."
sleep 15

# 건강성 체크
echo "🏥 건강성 체크..."
curl -f http://localhost:8501/_stcore/health || {
    echo "❌ UI 서버 건강성 체크 실패"
    exit 1
}

echo "✅ 배포 완료!"
```

---

## 📋 운영 체크리스트

### **일일 점검**
- [ ] 시스템 리소스 사용률 확인
- [ ] 에러 로그 검토
- [ ] 응답 시간 모니터링
- [ ] 백업 상태 확인

### **주간 점검**
- [ ] 성능 트렌드 분석
- [ ] 보안 업데이트 확인
- [ ] 디스크 사용량 점검
- [ ] 로그 정리

### **월간 점검**
- [ ] 전체 시스템 백업
- [ ] 성능 최적화 검토
- [ ] 보안 감사
- [ ] 용량 계획 업데이트

---

## 🎯 성능 최적화 팁

### **1. 데이터베이스 최적화**
```python
# 대용량 데이터 처리 최적화
from core.auto_data_profiler import get_auto_data_profiler

profiler = get_auto_data_profiler()
profiler.set_sampling_threshold(10000)  # 10K 이상 시 샘플링
profiler.enable_caching(True)  # 결과 캐싱 활성화
```

### **2. 메모리 최적화**
```python
# 메모리 사용량 모니터링
import psutil
import gc

def optimize_memory():
    """메모리 최적화"""
    # 가비지 컬렉션 강제 실행
    gc.collect()
    
    # 메모리 사용량 로깅
    memory_percent = psutil.virtual_memory().percent
    if memory_percent > 80:
        # 캐시 정리 등 최적화 작업 수행
        clear_caches()
```

### **3. 캐싱 전략**
```python
# Redis 캐싱 설정
import redis
import pickle

redis_client = redis.Redis(host='localhost', port=6379, db=0)

def cache_result(key: str, data: any, ttl: int = 3600):
    """결과 캐싱"""
    redis_client.setex(key, ttl, pickle.dumps(data))

def get_cached_result(key: str):
    """캐시된 결과 조회"""
    cached = redis_client.get(key)
    return pickle.loads(cached) if cached else None
```

---

## 🎉 배포 완료 확인

배포가 완료되면 다음 사항들을 확인하세요:

✅ **기능 테스트**
- [ ] UI 접근 가능
- [ ] 파일 업로드 작동
- [ ] A2A 에이전트 통신 정상
- [ ] 데이터 분석 결과 생성
- [ ] 로그 기록 정상

✅ **성능 테스트**
- [ ] 응답 시간 < 5초
- [ ] 메모리 사용량 < 80%
- [ ] CPU 사용량 < 70%
- [ ] 동시 사용자 지원

✅ **보안 확인**
- [ ] HTTPS 적용
- [ ] 인증 시스템 작동
- [ ] 방화벽 설정 완료
- [ ] API 키 안전 저장

---

**🎊 축하합니다! CherryAI Phase 1-4 시스템이 성공적으로 프로덕션에 배포되었습니다!** 