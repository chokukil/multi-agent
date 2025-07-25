# Cherry AI Streamlit Platform - Production Dockerfile
FROM python:3.11-slim as base

# 시스템 업데이트 및 필수 패키지 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /app

# Python 종속성 설치를 위한 requirements 파일 복사
COPY requirements.txt .
COPY pyproject.toml .

# Python 종속성 설치
RUN pip3 install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY . .

# 포트 8501 노출 (Streamlit 기본 포트)
EXPOSE 8501

# 헬스체크 설정
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Streamlit 설정
RUN mkdir -p ~/.streamlit/
RUN echo "\
[general]\n\
email = \"\"\n\
" > ~/.streamlit/credentials.toml

RUN echo "\
[server]\n\
headless = true\n\
enableCORS = false\n\
port = 8501\n\
address = 0.0.0.0\n\
maxUploadSize = 1000\n\
enableXsrfProtection = true\n\
\n\
[client]\n\
showErrorDetails = false\n\
\n\
[logger]\n\
level = \"info\"\n\
" > ~/.streamlit/config.toml

# 비루트 사용자 생성 및 권한 설정
RUN useradd -m -u 1000 streamlit && chown -R streamlit:streamlit /app
USER streamlit

# 애플리케이션 실행
ENTRYPOINT ["streamlit", "run", "cherry_ai_streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]