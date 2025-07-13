# 하드코딩 탐지 결과 리포트

**스캔 일시**: 20250713_211422

## 📊 전체 요약

- **스캔 파일 수**: 39159개
- **위반 발견**: 96144개
- **LLM First 준수도**: 23.4/100
- **문제 파일 수**: 10859개

## 🎯 위반 유형별 분포

- **rule_based_logic**: 23106개
- **pattern_matching**: 6446개
- **conditional_hardcode**: 28672개
- **dataset_dependency**: 515개
- **template_response**: 7374개
- **hardcoded_values**: 30000개
- **fixed_workflow**: 31개

## ⚠️ 심각도별 분포

- ⚠️ **high**: 52269개
- 🚨 **critical**: 515개
- ⚡ **medium**: 43360개

## 🔧 우선 리팩토링 대상

1. `.venv/lib/python3.11/site-packages/torch/testing/_internal/generated/annotated_fn_args.py`
2. `.venv/lib/python3.11/site-packages/pymdownx/twemoji_db.py`
3. `.venv/lib/python3.11/site-packages/faker/providers/job/de_AT/__init__.py`
4. `.venv/lib/python3.11/site-packages/pymdownx/gemoji_db.py`
5. `.venv/lib/python3.11/site-packages/pymdownx/emoji1_db.py`
6. `.venv/lib/python3.11/site-packages/h2o/estimators/xgboost.py`
7. `.venv/lib/python3.11/site-packages/sympy/parsing/autolev/_listener_autolev_antlr.py`
8. `.venv/lib/python3.11/site-packages/numpy/f2py/crackfortran.py`
9. `.venv/lib/python3.11/site-packages/setuptools/config/_validate_pyproject/fastjsonschema_validations.py`
10. `.venv/lib/python3.11/site-packages/torch/_export/serde/serialize.py`

## 📋 주요 위반 사례

### .venv/lib/python3.11/site-packages/IPython/__init__.py:129
- **유형**: dataset_dependency
- **설명**: 데이터셋 특화 하드코딩: 'survive'
- **코드**: `This is a public API method, and will survive implementation changes.`
- **제안**: 범용적 분석 로직으로 대체. 특정 데이터셋('survive')에 의존하지 않는 LLM 기반 분석 구현

### .venv/lib/python3.11/site-packages/altair/examples/normed_parallel_coordinates.py:11
- **유형**: dataset_dependency
- **설명**: 데이터셋 특화 하드코딩: 'Iris dataset'
- **코드**: `This example shows a modified parallel coordinates chart with the Iris dataset,`
- **제안**: 범용적 분석 로직으로 대체. 특정 데이터셋('Iris dataset')에 의존하지 않는 LLM 기반 분석 구현

### .venv/lib/python3.11/site-packages/altair/examples/parallel_coordinates.py:9
- **유형**: dataset_dependency
- **설명**: 데이터셋 특화 하드코딩: 'Iris dataset'
- **코드**: `This example shows a parallel coordinates chart with the Iris dataset.`
- **제안**: 범용적 분석 로직으로 대체. 특정 데이터셋('Iris dataset')에 의존하지 않는 LLM 기반 분석 구현

### .venv/lib/python3.11/site-packages/catboost/datasets.py:144
- **유형**: dataset_dependency
- **설명**: 데이터셋 특화 하드코딩: 'titanic'
- **코드**: `def titanic():`
- **제안**: 범용적 분석 로직으로 대체. 특정 데이터셋('titanic')에 의존하지 않는 LLM 기반 분석 구현

### .venv/lib/python3.11/site-packages/docutils/transforms/universal.py:15
- **유형**: dataset_dependency
- **설명**: 데이터셋 특화 하드코딩: 'pClass'
- **코드**: `- `StripClassesAndElements`: Remove elements with classes`
- **제안**: 범용적 분석 로직으로 대체. 특정 데이터셋('pClass')에 의존하지 않는 LLM 기반 분석 구현

