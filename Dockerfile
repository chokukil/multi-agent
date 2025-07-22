# Universal Engine Docker Container
# Multi-stage build for optimized production image

# Stage 1: Base Python environment
FROM python:3.11-slim as base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    netcat-openbsd \
    supervisor \
    nginx \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Stage 2: Dependencies installation
FROM base as dependencies

# Copy requirements first for better caching
COPY requirements.txt requirements-prod.txt ./

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r requirements-prod.txt

# Stage 3: Application
FROM dependencies as application

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/logs /app/pids /app/data /app/cache && \
    chown -R appuser:appuser /app

# Copy configuration files
COPY docker/supervisor.conf /etc/supervisor/conf.d/universal_engine.conf
COPY docker/nginx.conf /etc/nginx/sites-available/default

# Create startup script
RUN cat > /app/docker_start.sh << 'EOF'
#!/bin/bash
set -e

# Function to wait for service
wait_for_service() {
    local host=$1
    local port=$2
    local service=$3
    local max_attempts=30
    
    echo "Waiting for $service at $host:$port..."
    for i in $(seq 1 $max_attempts); do
        if nc -z $host $port; then
            echo "$service is ready!"
            return 0
        fi
        echo "Attempt $i/$max_attempts - waiting for $service..."
        sleep 2
    done
    
    echo "ERROR: $service failed to start"
    return 1
}

# Start services based on environment
echo "Starting Universal Engine in Docker..."

# Initialize environment
export PYTHONPATH=/app:$PYTHONPATH

# Start based on LLM provider
if [ "$LLM_PROVIDER" = "ollama" ]; then
    echo "Starting with Ollama (external service expected)"
    wait_for_service ${OLLAMA_HOST:-ollama} ${OLLAMA_PORT:-11434} "Ollama"
elif [ "$LLM_PROVIDER" = "openai" ]; then
    echo "Using OpenAI API"
    if [ -z "$OPENAI_API_KEY" ]; then
        echo "ERROR: OPENAI_API_KEY is required"
        exit 1
    fi
else
    echo "ERROR: Unsupported LLM_PROVIDER: $LLM_PROVIDER"
    exit 1
fi

# Initialize Universal Engine
echo "Initializing Universal Engine system..."
python -c "
import asyncio
from core.universal_engine.initialization.system_initializer import UniversalEngineInitializer

async def init():
    initializer = UniversalEngineInitializer()
    success = await initializer.initialize_system()
    if not success:
        print('Initialization failed')
        exit(1)
    print('System initialized successfully')

asyncio.run(init())
"

# Start supervisor to manage processes
exec supervisord -c /etc/supervisor/conf.d/universal_engine.conf -n
EOF

# Make startup script executable
RUN chmod +x /app/docker_start.sh

# Switch to app user
USER appuser

# Expose ports
EXPOSE 8000 8306 8307 8308 8309 8310 8311 8312 8313 8314 8315

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Environment variables
ENV PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LOG_LEVEL=INFO

# Default command
CMD ["/app/docker_start.sh"]