#!/bin/bash

# 🦙 Ollama 환경변수 자동 설정 스크립트
# CherryAI - Ollama Tool Calling 최적화

echo "🦙 CherryAI Ollama 환경 설정 시작..."

# 기본 환경변수 설정
echo "📝 환경변수 설정 중..."

# LLM 제공자를 Ollama로 설정
export LLM_PROVIDER=OLLAMA

# 권장 모델 설정 (tool calling 지원)
export OLLAMA_MODEL=llama3.1:8b

# Ollama 서버 URL
export OLLAMA_BASE_URL=http://localhost:11434

# Ollama 전용 타임아웃 (10분)
export OLLAMA_TIMEOUT=600

# 자동 모델 전환 활성화
export OLLAMA_AUTO_SWITCH_MODEL=true

# .env 파일에 저장
echo "💾 .env 파일에 설정 저장 중..."

cat >> .env << EOF

# 🦙 Ollama 설정 (자동 생성)
LLM_PROVIDER=OLLAMA
OLLAMA_MODEL=llama3.1:8b
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_TIMEOUT=600
OLLAMA_AUTO_SWITCH_MODEL=true

EOF

echo "✅ 환경변수 설정 완료!"

# Ollama 서버 상태 확인
echo "🔍 Ollama 서버 상태 확인 중..."

if curl -s http://localhost:11434/api/version > /dev/null; then
    echo "✅ Ollama 서버가 실행 중입니다."
    
    # 사용 가능한 모델 확인
    echo "📋 설치된 모델 확인 중..."
    models=$(curl -s http://localhost:11434/api/tags | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    models = [m['name'] for m in data.get('models', [])]
    print(f'설치된 모델: {len(models)}개')
    for model in models:
        print(f'  - {model}')
except:
    print('모델 정보를 가져올 수 없습니다.')
")
    echo "$models"
    
    # 권장 모델이 설치되어 있는지 확인
    if curl -s http://localhost:11434/api/tags | grep -q "llama3.1:8b"; then
        echo "✅ 권장 모델 llama3.1:8b가 설치되어 있습니다."
    else
        echo "⚠️ 권장 모델 llama3.1:8b가 설치되어 있지 않습니다."
        echo "💡 다음 명령어로 설치하세요:"
        echo "   ollama pull llama3.1:8b"
    fi
    
else
    echo "❌ Ollama 서버가 실행되지 않고 있습니다."
    echo "💡 다음 명령어로 Ollama를 시작하세요:"
    echo "   ollama serve"
fi

# 시스템 RAM 확인 및 권장사항
echo "🖥️ 시스템 리소스 확인 중..."

if command -v python3 > /dev/null; then
    python3 -c "
import psutil
ram_gb = psutil.virtual_memory().total // (1024**3)
print(f'시스템 RAM: {ram_gb}GB')

recommendations = {
    'light': ('qwen2.5:3b', '4GB RAM', '가벼운 작업용'),
    'balanced': ('llama3.1:8b', '10GB RAM', '균형잡힌 성능'), 
    'powerful': ('qwen2.5:14b', '16GB RAM', '고성능 작업'),
    'coding': ('qwen2.5-coder:7b', '9GB RAM', '코딩 전문')
}

if ram_gb >= 16:
    rec = recommendations['powerful']
    print(f'💪 고성능 시스템 감지! 권장 모델: {rec[0]} ({rec[2]})')
elif ram_gb >= 10:
    rec = recommendations['balanced']
    print(f'⚖️ 균형 시스템 감지! 권장 모델: {rec[0]} ({rec[2]})')
elif ram_gb >= 6:
    rec = recommendations['light']
    print(f'🪶 경량 시스템 감지! 권장 모델: {rec[0]} ({rec[2]})')
else:
    print(f'⚠️ RAM이 부족합니다 ({ram_gb}GB). 최소 6GB 권장')
    print('   경량 모델: qwen2.5:0.5b (2GB RAM)')
"
fi

echo ""
echo "🎉 Ollama 환경 설정이 완료되었습니다!"
echo ""
echo "📋 다음 단계:"
echo "1. Ollama 서버 시작: ollama serve"
echo "2. 권장 모델 다운로드: ollama pull llama3.1:8b"
echo "3. CherryAI 재시작하여 새 설정 적용"
echo ""
echo "🔧 Tool Calling 지원 모델들:"
echo "  - llama3.1:8b (균형, 8GB RAM)"
echo "  - qwen2.5:7b (빠름, 8GB RAM)"  
echo "  - qwen2.5-coder:7b (코딩 전문, 9GB RAM)"
echo "  - mistral:7b (추론 우수, 8GB RAM)"
echo ""
echo "💡 모델 변경은 환경변수 OLLAMA_MODEL을 수정하세요." 