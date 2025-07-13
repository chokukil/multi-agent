#!/bin/bash

# CherryAI 캐시 정리 스크립트
# 하위폴더까지 재귀적으로 Python 캐시 파일들을 제거합니다

echo "🍒 CherryAI 캐시 정리 시작..."

# 현재 디렉토리 출력
echo "📂 현재 디렉토리: $(pwd)"

# __pycache__ 폴더들 제거
echo "🗑️  __pycache__ 폴더들 제거 중..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# .pyc 파일들 제거
echo "🗑️  .pyc 파일들 제거 중..."
find . -type f -name "*.pyc" -delete 2>/dev/null || true

# .pyo 파일들 제거
echo "🗑️  .pyo 파일들 제거 중..."
find . -type f -name "*.pyo" -delete 2>/dev/null || true

# pytest 캐시 제거
echo "🗑️  pytest 캐시 제거 중..."
find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true

# coverage 파일들 제거
echo "🗑️  coverage 파일들 제거 중..."
find . -type f -name ".coverage" -delete 2>/dev/null || true
find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true

# mypy 캐시 제거
echo "🗑️  mypy 캐시 제거 중..."
find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true

# build 폴더들 제거
echo "🗑️  build 폴더들 제거 중..."
find . -type d -name "build" -exec rm -rf {} + 2>/dev/null || true

# dist 폴더들 제거
echo "🗑️  dist 폴더들 제거 중..."
find . -type d -name "dist" -exec rm -rf {} + 2>/dev/null || true

# .egg-info 폴더들 제거
echo "🗑️  .egg-info 폴더들 제거 중..."
find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true

# .tox 폴더들 제거
echo "🗑️  .tox 폴더들 제거 중..."
find . -type d -name ".tox" -exec rm -rf {} + 2>/dev/null || true

# Jupyter 체크포인트 제거
echo "🗑️  Jupyter 체크포인트 제거 중..."
find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true

# VS Code 설정 (선택적)
echo "🗑️  VS Code 캐시 제거 중..."
find . -type d -name ".vscode" -path "*/__pycache__*" -exec rm -rf {} + 2>/dev/null || true

echo "✅ 캐시 정리 완료!"
echo "🎉 모든 Python 캐시 파일과 폴더가 재귀적으로 제거되었습니다." 