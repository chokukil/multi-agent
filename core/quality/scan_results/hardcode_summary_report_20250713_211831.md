# 하드코딩 탐지 결과 리포트

**스캔 일시**: 20250713_211831

## 📊 전체 요약

- **스캔 파일 수**: 7개
- **위반 발견**: 49개
- **LLM First 준수도**: 29.7/100
- **문제 파일 수**: 3개

## 🎯 위반 유형별 분포

- **rule_based_logic**: 6개
- **template_response**: 2개
- **hardcoded_values**: 33개
- **conditional_hardcode**: 8개

## ⚠️ 심각도별 분포

- ⚠️ **high**: 14개
- ⚡ **medium**: 35개

## 🔧 우선 리팩토링 대상

1. `tests/production_ai_agent_validation.py`
2. `tests/comprehensive_ai_agent_validation.py`
3. `tests/simple_workflow_test.py`

## 📋 주요 위반 사례

