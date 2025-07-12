# 🔧 CherryAI 데이터 시스템 통합 솔루션

## 🚨 현재 문제점

### 1. 분산된 데이터 시스템
- **A2A 시스템**: 포트 8100, 8306-8314 (9개 서버)
- **Standalone 서버**: 포트 8080 (Flask 기반)
- **독립 Pandas Agent**: 포트 8315 (격리된 실행)

### 2. 데이터 접근 불일치
```
❌ 서버별 별도 데이터 저장소
❌ 세션 데이터 미공유
❌ 파일 접근 경합 발생
❌ 인코딩 오류 (Excel → UTF-8)
```

## 💡 해결 방안

### Phase 1: 데이터 레이어 통합
```python
# 통합 데이터 매니저
class UnifiedDataManager:
    def __init__(self):
        self.shared_storage = SharedDataStore()
        self.session_manager = GlobalSessionManager()
        self.file_cache = SmartFileCache()
    
    async def load_excel_safely(self, file_path: str):
        """안전한 Excel 파일 로딩"""
        try:
            # 파일 락 처리
            with FileLock(file_path):
                df = pd.read_excel(file_path, engine='openpyxl')
                return df
        except UnicodeDecodeError:
            # 인코딩 문제 해결
            df = pd.read_excel(file_path, engine='xlrd')
            return df
```

### Phase 2: API 게이트웨이 구축
```
┌─────────────────┐
│   API Gateway   │  ← 모든 요청의 단일 진입점
│   (Port 8000)   │
└─────────────────┘
         │
┌─────────────────┐
│ Unified Data    │  ← 통합 데이터 매니저
│ Manager         │
└─────────────────┘
         │
┌─────────────────┐
│ Backend Services│  ← 기존 A2A 에이전트들
│ (8306-8314)     │
└─────────────────┘
```

### Phase 3: 세션 동기화
```python
# 글로벌 세션 상태
class GlobalSession:
    def __init__(self):
        self.data_registry = {}
        self.active_connections = {}
    
    async def sync_data_across_services(self, session_id: str):
        """모든 서비스에 세션 데이터 동기화"""
        session_data = self.get_session(session_id)
        for service in self.active_connections:
            await service.update_session_data(session_data)
```

## 🎯 즉시 실행 가능한 해결책

### 1. 올바른 서버 사용
```bash
# ❌ 잘못된 방법 - 독립 실행
python a2a_ds_servers/pandas_agent/pandas_agent_server.py 8315

# ✅ 올바른 방법 - 통합 시스템 사용
./ai_ds_team_system_start.sh  # 전체 시스템 시작
curl -X POST http://localhost:8080/api/sample-data  # 테스트 데이터 생성
```

### 2. 데이터 직접 로딩 테스트
```python
import pandas as pd

# 안전한 Excel 로딩
def safe_excel_load(file_path):
    try:
        return pd.read_excel(file_path, engine='openpyxl')
    except Exception as e:
        print(f"openpyxl 실패: {e}")
        try:
            return pd.read_excel(file_path, engine='xlrd')
        except Exception as e2:
            print(f"xlrd도 실패: {e2}")
            return None

df = safe_excel_load('a2a_ds_servers/artifacts/data/shared_dataframes/session_9bf4ad1b_ion_implant_3lot_dataset.xlsx')
print(f"로딩 결과: {df.shape if df is not None else 'None'}")
```

### 3. 파일 권한 및 락 문제 해결
```bash
# 파일 권한 확인
ls -la a2a_ds_servers/artifacts/data/shared_dataframes/session_9bf4ad1b_ion_implant_3lot_dataset.xlsx

# 파일 사용 중인 프로세스 확인
lsof a2a_ds_servers/artifacts/data/shared_dataframes/session_9bf4ad1b_ion_implant_3lot_dataset.xlsx
```

## 📋 Action Items

### 긴급 (당장 해결)
- [ ] 독립 8315 서버 사용 중단
- [ ] Standalone 서버 (8080) 업로드 API 수정
- [ ] Excel 파일 인코딩 처리 개선

### 중요 (이번 주)
- [ ] 통합 데이터 매니저 구현
- [ ] API 게이트웨이 설계
- [ ] 세션 동기화 로직 구현

### 장기 (다음 버전)
- [ ] 마이크로서비스 아키텍처 재설계
- [ ] Redis 기반 세션 스토어 도입
- [ ] 분산 캐시 시스템 구축 