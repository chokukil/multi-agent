# 하드코딩 탐지 결과 리포트

**스캔 일시**: 20250713_211830

## 📊 전체 요약

- **스캔 파일 수**: 127개
- **위반 발견**: 1640개
- **LLM First 준수도**: 28.1/100
- **문제 파일 수**: 105개

## 🎯 위반 유형별 분포

- **rule_based_logic**: 268개
- **template_response**: 72개
- **conditional_hardcode**: 238개
- **hardcoded_values**: 950개
- **pattern_matching**: 108개
- **dataset_dependency**: 1개
- **fixed_workflow**: 3개

## ⚠️ 심각도별 분포

- ⚠️ **high**: 567개
- ⚡ **medium**: 1072개
- 🚨 **critical**: 1개

## 🔧 우선 리팩토링 대상

1. `core/a2a/a2a_streamlit_client.py`
2. `core/app_components/data_workspace.py`
3. `core/query_processing/final_answer_structuring.py`
4. `core/advanced_code_tracker.py`
5. `core/specialized_data_agents.py`
6. `core/utils/streaming.py`
7. `core/query_processing/answer_predictor.py`
8. `core/monitoring/a2a_performance_profiler.py`
9. `core/streaming/streaming_orchestrator.py`
10. `core/system_health_checker.py`

## 📋 주요 위반 사례

### core/direct_analysis.py:242
- **유형**: dataset_dependency
- **설명**: 데이터셋 특화 하드코딩: 'pclass'
- **코드**: `for pclass, rate in class_survival.items():`
- **제안**: 범용적 분석 로직으로 대체. 특정 데이터셋('pclass')에 의존하지 않는 LLM 기반 분석 구현

