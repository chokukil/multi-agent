# 🍒 CherryAI 캐시 정리 스크립트

CherryAI 프로젝트에서 Python 캐시 파일들을 하위폴더까지 재귀적으로 제거하는 스크립트들입니다.

## 📁 파일 구성

- `clear_cache.sh` - Shell 스크립트 버전 (빠른 실행)
- `clear_cache.py` - Python 스크립트 버전 (고급 기능)
- `README_CACHE_SCRIPTS.md` - 사용법 안내서 (이 파일)

## 🚀 빠른 시작

### Shell 스크립트 사용 (추천)
```bash
# 간단한 캐시 정리
./clear_cache.sh
```

### Python 스크립트 사용 (고급 기능)
```bash
# 기본 캐시 정리
python clear_cache.py

# DRY RUN (실제 삭제 없이 미리보기)
python clear_cache.py --dry-run

# 상세 출력 포함
python clear_cache.py --verbose

# 특정 경로 정리
python clear_cache.py --path /path/to/directory
```

## 🛡️ 안전 기능

### 가상환경 보호
- `.venv`, `venv`, `.env`, `env` 폴더는 **자동으로 제외**
- 가상환경 캐시까지 정리하려면 `--include-venv` 옵션 사용

### DRY RUN 모드
```bash
# 실제 삭제 없이 미리보기
python clear_cache.py --dry-run
```

## 🗑️ 제거 대상 파일/폴더

### 폴더
- `__pycache__` - Python 바이트코드 캐시
- `.pytest_cache` - pytest 캐시
- `.mypy_cache` - mypy 타입 체크 캐시
- `build` - 빌드 아티팩트
- `dist` - 배포 파일
- `*.egg-info` - 설치 정보
- `.tox` - tox 테스트 환경
- `.ipynb_checkpoints` - Jupyter 체크포인트
- `htmlcov` - 커버리지 HTML 리포트

### 파일
- `*.pyc` - Python 바이트코드 파일
- `*.pyo` - Python 최적화 바이트코드
- `.coverage` - 커버리지 데이터
- `*.log` - 로그 파일
- `.DS_Store` - macOS 메타데이터 파일

## 📊 사용 예시

### 기본 사용
```bash
# Shell 스크립트 (가장 간단)
./clear_cache.sh

# Python 스크립트 (기본)
python clear_cache.py
```

### 고급 사용
```bash
# 특정 폴더만 정리
python clear_cache.py --path ./a2a_ds_servers

# 삭제 전 미리보기
python clear_cache.py --dry-run --verbose

# 가상환경 포함 정리 (주의!)
python clear_cache.py --include-venv
```

## 🎯 실행 결과 예시

```
🍒 CherryAI 캐시 정리 시작...
📂 대상 디렉토리: /Users/gukil/CherryAI/CherryAI_0623
🚫 가상환경 디렉토리(.venv, venv 등)는 제외됩니다.
🔍 캐시 파일과 폴더 검색 중...
📊 총 77개의 캐시 아이템을 발견했습니다.

🗑️  캐시 제거 중...
🗑️  파일 제거: /Users/gukil/CherryAI/CherryAI_0623/.DS_Store
🗑️  폴더 제거: /Users/gukil/CherryAI/CherryAI_0623/.pytest_cache
...

✅ 캐시 정리 완료!
📊 제거된 파일: 55개
📊 제거된 폴더: 22개
🎉 모든 Python 캐시 파일과 폴더가 재귀적으로 제거되었습니다.
```

## 🔧 Python 스크립트 옵션

| 옵션 | 단축형 | 설명 |
|------|--------|------|
| `--path` | `-p` | 정리할 경로 지정 (기본값: 현재 디렉토리) |
| `--dry-run` | `-d` | 실제 삭제 없이 미리보기 |
| `--verbose` | `-v` | 상세한 출력 |
| `--include-venv` | - | 가상환경 디렉토리도 포함 |

## 📋 CherryAI 시스템 통합 [[memory:40974]]

CherryAI 시스템 테스트 시 권장 절차:

1. **캐시 정리**
   ```bash
   ./clear_cache.sh
   # 또는
   python clear_cache.py
   ```

2. **기존 프로세스 종료**
   ```bash
   pkill -f streamlit && pkill -f python
   ```

3. **A2A 서버 상태 확인**
   ```bash
   ps aux | grep python
   ```

4. **서버 시작**
   ```bash
   ./ai_ds_team_system_start.sh
   ```

5. **UI 실행**
   ```bash
   streamlit run main.py
   ```

## ⚠️ 주의사항

- **가상환경 안전**: 기본적으로 `.venv`, `venv` 등은 제외됩니다
- **백업 권장**: 중요한 로그가 있다면 실행 전 백업하세요
- **DRY RUN 활용**: 처음 사용 시 `--dry-run`으로 미리 확인하세요

## 🔗 관련 스크립트

- `ai_ds_team_system_start.sh` - 시스템 시작
- `ai_ds_team_system_stop.sh` - 시스템 종료
- `quick_start.bat` - Windows 빠른 시작

---

🍒 **CherryAI 프로젝트** - 세계 최초 A2A + MCP 통합 플랫폼 