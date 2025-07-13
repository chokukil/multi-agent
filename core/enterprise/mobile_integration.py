"""
Cross-Platform Mobile Integration
Phase 4.5: 모바일 및 크로스 플랫폼

핵심 기능:
- React Native 모바일 앱 지원
- 오프라인 분석 기능
- 음성 쿼리 지원
- Electron 데스크톱 앱
- 실시간 알림 시스템
- 모바일 최적화 API
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union, Callable
from enum import Enum
from pathlib import Path
import sqlite3
import base64
import tempfile

logger = logging.getLogger(__name__)

class PlatformType(Enum):
    """플랫폼 유형"""
    REACT_NATIVE = "react_native"
    ELECTRON = "electron"
    PWA = "pwa"
    WEB = "web"

class NotificationType(Enum):
    """알림 유형"""
    ANALYSIS_COMPLETE = "analysis_complete"
    INSIGHT_GENERATED = "insight_generated"
    SYSTEM_ALERT = "system_alert"
    UPDATE_AVAILABLE = "update_available"

class OfflineJobStatus(Enum):
    """오프라인 작업 상태"""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SYNCING = "syncing"

@dataclass
class MobileAPIRequest:
    """모바일 API 요청"""
    request_id: str
    platform: PlatformType
    device_id: str
    user_id: str
    data: Dict[str, Any]
    offline_capable: bool = True
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class NotificationPayload:
    """알림 페이로드"""
    notification_id: str
    notification_type: NotificationType
    title: str
    message: str
    data: Dict[str, Any]
    platform: PlatformType
    device_id: str
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class OfflineJob:
    """오프라인 작업"""
    job_id: str
    device_id: str
    job_type: str
    data: Dict[str, Any]
    status: OfflineJobStatus
    created_at: datetime
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

@dataclass
class VoiceQuery:
    """음성 쿼리"""
    query_id: str
    device_id: str
    audio_data: str  # base64 encoded
    transcribed_text: Optional[str] = None
    analysis_result: Optional[Dict[str, Any]] = None
    created_at: datetime = field(default_factory=datetime.now)

class MobileDatabase:
    """모바일 데이터베이스 관리"""
    
    def __init__(self, db_path: str = "core/enterprise/mobile.db"):
        self.db_path = db_path
        self._initialize_database()
    
    def _initialize_database(self):
        """데이터베이스 초기화"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            # 모바일 기기 등록
            conn.execute("""
                CREATE TABLE IF NOT EXISTS mobile_devices (
                    device_id TEXT PRIMARY KEY,
                    platform TEXT NOT NULL,
                    device_info TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    push_token TEXT,
                    is_active BOOLEAN DEFAULT TRUE,
                    last_sync TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # 오프라인 작업 큐
            conn.execute("""
                CREATE TABLE IF NOT EXISTS offline_jobs (
                    job_id TEXT PRIMARY KEY,
                    device_id TEXT NOT NULL,
                    job_type TEXT NOT NULL,
                    data TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP,
                    result TEXT,
                    error_message TEXT,
                    FOREIGN KEY (device_id) REFERENCES mobile_devices (device_id)
                )
            """)
            
            # 알림 기록
            conn.execute("""
                CREATE TABLE IF NOT EXISTS notifications (
                    notification_id TEXT PRIMARY KEY,
                    notification_type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    message TEXT NOT NULL,
                    data TEXT NOT NULL,
                    platform TEXT NOT NULL,
                    device_id TEXT NOT NULL,
                    is_read BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (device_id) REFERENCES mobile_devices (device_id)
                )
            """)
            
            # 음성 쿼리 기록
            conn.execute("""
                CREATE TABLE IF NOT EXISTS voice_queries (
                    query_id TEXT PRIMARY KEY,
                    device_id TEXT NOT NULL,
                    audio_data TEXT NOT NULL,
                    transcribed_text TEXT,
                    analysis_result TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (device_id) REFERENCES mobile_devices (device_id)
                )
            """)
    
    def register_device(self, device_id: str, platform: PlatformType, device_info: Dict[str, Any], user_id: str) -> bool:
        """모바일 기기 등록"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO mobile_devices 
                    (device_id, platform, device_info, user_id, is_active)
                    VALUES (?, ?, ?, ?, ?)
                """, (device_id, platform.value, json.dumps(device_info), user_id, True))
            return True
        except Exception as e:
            logger.error(f"기기 등록 실패: {e}")
            return False
    
    def save_offline_job(self, job: OfflineJob) -> bool:
        """오프라인 작업 저장"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO offline_jobs 
                    (job_id, device_id, job_type, data, status, created_at, completed_at, result, error_message)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    job.job_id, job.device_id, job.job_type, json.dumps(job.data),
                    job.status.value, job.created_at.isoformat(),
                    job.completed_at.isoformat() if job.completed_at else None,
                    json.dumps(job.result) if job.result else None,
                    job.error_message
                ))
            return True
        except Exception as e:
            logger.error(f"오프라인 작업 저장 실패: {e}")
            return False
    
    def get_offline_jobs(self, device_id: str, status: Optional[OfflineJobStatus] = None) -> List[OfflineJob]:
        """오프라인 작업 조회"""
        jobs = []
        try:
            with sqlite3.connect(self.db_path) as conn:
                if status:
                    cursor = conn.execute("""
                        SELECT * FROM offline_jobs 
                        WHERE device_id = ? AND status = ?
                        ORDER BY created_at DESC
                    """, (device_id, status.value))
                else:
                    cursor = conn.execute("""
                        SELECT * FROM offline_jobs 
                        WHERE device_id = ?
                        ORDER BY created_at DESC
                    """, (device_id,))
                
                for row in cursor.fetchall():
                    job = OfflineJob(
                        job_id=row[0],
                        device_id=row[1],
                        job_type=row[2],
                        data=json.loads(row[3]),
                        status=OfflineJobStatus(row[4]),
                        created_at=datetime.fromisoformat(row[5]),
                        completed_at=datetime.fromisoformat(row[6]) if row[6] else None,
                        result=json.loads(row[7]) if row[7] else None,
                        error_message=row[8]
                    )
                    jobs.append(job)
        except Exception as e:
            logger.error(f"오프라인 작업 조회 실패: {e}")
        
        return jobs

class OfflineAnalysisEngine:
    """오프라인 분석 엔진"""
    
    def __init__(self, mobile_db: MobileDatabase):
        self.mobile_db = mobile_db
        self.temp_dir = Path(tempfile.gettempdir()) / "cherry_mobile"
        self.temp_dir.mkdir(exist_ok=True)
    
    async def queue_offline_job(self, device_id: str, job_type: str, data: Dict[str, Any]) -> str:
        """오프라인 작업 큐에 추가"""
        job_id = f"job_{int(time.time())}_{device_id}"
        
        job = OfflineJob(
            job_id=job_id,
            device_id=device_id,
            job_type=job_type,
            data=data,
            status=OfflineJobStatus.QUEUED,
            created_at=datetime.now()
        )
        
        success = self.mobile_db.save_offline_job(job)
        if success:
            # 백그라운드에서 작업 처리 시작
            asyncio.create_task(self._process_offline_job(job))
            return job_id
        else:
            raise Exception("Failed to queue offline job")
    
    async def _process_offline_job(self, job: OfflineJob):
        """오프라인 작업 처리"""
        try:
            # 상태를 처리 중으로 변경
            job.status = OfflineJobStatus.PROCESSING
            self.mobile_db.save_offline_job(job)
            
            # 작업 유형에 따른 처리
            if job.job_type == "data_analysis":
                result = await self._perform_offline_analysis(job.data)
            elif job.job_type == "voice_transcription":
                result = await self._perform_voice_transcription(job.data)
            elif job.job_type == "data_export":
                result = await self._perform_data_export(job.data)
            else:
                raise Exception(f"Unknown job type: {job.job_type}")
            
            # 작업 완료
            job.status = OfflineJobStatus.COMPLETED
            job.completed_at = datetime.now()
            job.result = result
            self.mobile_db.save_offline_job(job)
            
            logger.info(f"오프라인 작업 완료: {job.job_id}")
            
        except Exception as e:
            # 작업 실패
            job.status = OfflineJobStatus.FAILED
            job.completed_at = datetime.now()
            job.error_message = str(e)
            self.mobile_db.save_offline_job(job)
            
            logger.error(f"오프라인 작업 실패: {job.job_id}, {e}")
    
    async def _perform_offline_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """오프라인 데이터 분석"""
        import pandas as pd
        
        # 데이터 로드
        if "csv_data" in data:
            df = pd.read_csv(data["csv_data"])
        elif "json_data" in data:
            df = pd.DataFrame(data["json_data"])
        else:
            raise Exception("No valid data provided")
        
        # 기본 분석 수행
        analysis_result = {
            "dataset_info": {
                "shape": df.shape,
                "columns": list(df.columns),
                "dtypes": df.dtypes.to_dict()
            },
            "basic_stats": df.describe().to_dict() if len(df.select_dtypes(include=['number']).columns) > 0 else {},
            "missing_values": df.isnull().sum().to_dict(),
            "duplicate_rows": int(df.duplicated().sum()),
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        return analysis_result
    
    async def _perform_voice_transcription(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """음성 전사 (모의 구현)"""
        # 실제 환경에서는 speech-to-text API 사용
        audio_data = data.get("audio_data", "")
        
        # 모의 전사 결과
        transcribed_text = "분석해주세요"  # 실제로는 음성 인식 결과
        
        return {
            "transcribed_text": transcribed_text,
            "confidence": 0.95,
            "language": "ko-KR",
            "processing_time_ms": 1500
        }
    
    async def _perform_data_export(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """데이터 내보내기"""
        export_format = data.get("format", "csv")
        export_data = data.get("data", [])
        
        # 임시 파일 생성
        timestamp = int(time.time())
        filename = f"export_{timestamp}.{export_format}"
        file_path = self.temp_dir / filename
        
        import pandas as pd
        df = pd.DataFrame(export_data)
        
        if export_format == "csv":
            df.to_csv(file_path, index=False)
        elif export_format == "json":
            df.to_json(file_path, orient="records")
        elif export_format == "xlsx":
            df.to_excel(file_path, index=False)
        
        return {
            "file_path": str(file_path),
            "filename": filename,
            "file_size": file_path.stat().st_size,
            "export_format": export_format,
            "exported_rows": len(df)
        }

class VoiceQueryProcessor:
    """음성 쿼리 처리기"""
    
    def __init__(self, mobile_db: MobileDatabase):
        self.mobile_db = mobile_db
    
    async def process_voice_query(self, device_id: str, audio_data: str) -> str:
        """음성 쿼리 처리"""
        query_id = f"voice_{int(time.time())}_{device_id}"
        
        # 음성 쿼리 저장
        with sqlite3.connect(self.mobile_db.db_path) as conn:
            conn.execute("""
                INSERT INTO voice_queries (query_id, device_id, audio_data, created_at)
                VALUES (?, ?, ?, ?)
            """, (query_id, device_id, audio_data, datetime.now().isoformat()))
        
        try:
            # 음성 전사 (모의 구현)
            transcribed_text = await self._transcribe_audio(audio_data)
            
            # 전사된 텍스트로 분석 수행
            analysis_result = await self._analyze_text_query(transcribed_text)
            
            # 결과 업데이트
            with sqlite3.connect(self.mobile_db.db_path) as conn:
                conn.execute("""
                    UPDATE voice_queries 
                    SET transcribed_text = ?, analysis_result = ?
                    WHERE query_id = ?
                """, (transcribed_text, json.dumps(analysis_result), query_id))
            
            return query_id
            
        except Exception as e:
            logger.error(f"음성 쿼리 처리 실패: {e}")
            raise
    
    async def _transcribe_audio(self, audio_data: str) -> str:
        """음성 전사 (모의 구현)"""
        # 실제 환경에서는 Google Speech-to-Text, Azure Speech Services 등 사용
        # 여기서는 모의 결과 반환
        sample_queries = [
            "매출 데이터를 분석해주세요",
            "고객 만족도 트렌드를 보여주세요", 
            "이상치를 찾아주세요",
            "예측 분석을 실행해주세요",
            "보고서를 생성해주세요"
        ]
        
        import random
        return random.choice(sample_queries)
    
    async def _analyze_text_query(self, text: str) -> Dict[str, Any]:
        """텍스트 쿼리 분석"""
        # 간단한 키워드 기반 분석
        keywords = text.lower().split()
        
        analysis_type = "general"
        if any(keyword in keywords for keyword in ["매출", "revenue", "sales"]):
            analysis_type = "revenue_analysis"
        elif any(keyword in keywords for keyword in ["고객", "customer", "만족도"]):
            analysis_type = "customer_analysis"
        elif any(keyword in keywords for keyword in ["이상치", "anomaly", "outlier"]):
            analysis_type = "anomaly_detection"
        elif any(keyword in keywords for keyword in ["예측", "prediction", "forecast"]):
            analysis_type = "predictive_analysis"
        
        return {
            "analysis_type": analysis_type,
            "confidence": 0.85,
            "keywords": keywords,
            "suggested_actions": [
                f"{analysis_type} 실행",
                "데이터 로드",
                "결과 시각화"
            ]
        }

class PushNotificationManager:
    """푸시 알림 관리"""
    
    def __init__(self, mobile_db: MobileDatabase):
        self.mobile_db = mobile_db
    
    async def send_notification(self, payload: NotificationPayload) -> bool:
        """푸시 알림 전송"""
        try:
            # 알림 저장
            with sqlite3.connect(self.mobile_db.db_path) as conn:
                conn.execute("""
                    INSERT INTO notifications 
                    (notification_id, notification_type, title, message, data, platform, device_id, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    payload.notification_id,
                    payload.notification_type.value,
                    payload.title,
                    payload.message,
                    json.dumps(payload.data),
                    payload.platform.value,
                    payload.device_id,
                    payload.created_at.isoformat()
                ))
            
            # 실제 푸시 알림 전송 (모의 구현)
            await self._send_platform_notification(payload)
            
            logger.info(f"푸시 알림 전송 완료: {payload.notification_id}")
            return True
            
        except Exception as e:
            logger.error(f"푸시 알림 전송 실패: {e}")
            return False
    
    async def _send_platform_notification(self, payload: NotificationPayload):
        """플랫폼별 알림 전송"""
        if payload.platform == PlatformType.REACT_NATIVE:
            # Firebase Cloud Messaging 또는 Apple Push Notification Service
            logger.info(f"📱 React Native 푸시 알림: {payload.title}")
        elif payload.platform == PlatformType.ELECTRON:
            # Electron 네이티브 알림
            logger.info(f"🖥️ Electron 알림: {payload.title}")
        elif payload.platform == PlatformType.WEB:
            # Web Push API
            logger.info(f"🌐 웹 푸시 알림: {payload.title}")
        
        # 실제 구현에서는 각 플랫폼의 알림 서비스 API 호출

class MobileOptimizedAPI:
    """모바일 최적화 API"""
    
    def __init__(self):
        self.mobile_db = MobileDatabase()
        self.offline_engine = OfflineAnalysisEngine(self.mobile_db)
        self.voice_processor = VoiceQueryProcessor(self.mobile_db)
        self.notification_manager = PushNotificationManager(self.mobile_db)
    
    async def register_device(self, device_id: str, platform: PlatformType, 
                            device_info: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        """모바일 기기 등록"""
        success = self.mobile_db.register_device(device_id, platform, device_info, user_id)
        
        if success:
            # 환영 알림 전송
            welcome_notification = NotificationPayload(
                notification_id=f"welcome_{device_id}",
                notification_type=NotificationType.SYSTEM_ALERT,
                title="CherryAI에 오신 것을 환영합니다!",
                message="모바일에서도 강력한 AI 데이터 분석을 경험하세요.",
                data={"welcome": True},
                platform=platform,
                device_id=device_id
            )
            
            await self.notification_manager.send_notification(welcome_notification)
            
            return {"status": "success", "device_id": device_id}
        else:
            return {"status": "error", "message": "Device registration failed"}
    
    async def submit_offline_analysis(self, device_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """오프라인 분석 요청"""
        try:
            job_id = await self.offline_engine.queue_offline_job(device_id, "data_analysis", data)
            return {"status": "success", "job_id": job_id}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    async def get_offline_jobs(self, device_id: str) -> Dict[str, Any]:
        """오프라인 작업 상태 조회"""
        jobs = self.mobile_db.get_offline_jobs(device_id)
        
        job_list = []
        for job in jobs:
            job_dict = {
                "job_id": job.job_id,
                "job_type": job.job_type,
                "status": job.status.value,
                "created_at": job.created_at.isoformat(),
                "completed_at": job.completed_at.isoformat() if job.completed_at else None
            }
            
            if job.result:
                job_dict["result"] = job.result
            if job.error_message:
                job_dict["error_message"] = job.error_message
            
            job_list.append(job_dict)
        
        return {"status": "success", "jobs": job_list}
    
    async def process_voice_query(self, device_id: str, audio_data: str) -> Dict[str, Any]:
        """음성 쿼리 처리"""
        try:
            query_id = await self.voice_processor.process_voice_query(device_id, audio_data)
            return {"status": "success", "query_id": query_id}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    async def get_mobile_dashboard(self, device_id: str) -> Dict[str, Any]:
        """모바일 대시보드 데이터"""
        # 최근 작업들
        recent_jobs = self.mobile_db.get_offline_jobs(device_id, None)[:5]
        
        # 알림 개수
        with sqlite3.connect(self.mobile_db.db_path) as conn:
            cursor = conn.execute("""
                SELECT COUNT(*) FROM notifications 
                WHERE device_id = ? AND is_read = FALSE
            """, (device_id,))
            unread_notifications = cursor.fetchone()[0]
        
        # 통계
        total_jobs = len(recent_jobs)
        completed_jobs = len([job for job in recent_jobs if job.status == OfflineJobStatus.COMPLETED])
        
        return {
            "status": "success",
            "dashboard": {
                "total_jobs": total_jobs,
                "completed_jobs": completed_jobs,
                "success_rate": (completed_jobs / total_jobs * 100) if total_jobs > 0 else 0,
                "unread_notifications": unread_notifications,
                "recent_jobs": [
                    {
                        "job_id": job.job_id,
                        "job_type": job.job_type,
                        "status": job.status.value,
                        "created_at": job.created_at.isoformat()
                    }
                    for job in recent_jobs
                ],
                "device_id": device_id,
                "last_updated": datetime.now().isoformat()
            }
        }

# 앱 설정 생성기
class AppConfigGenerator:
    """모바일/데스크톱 앱 설정 생성"""
    
    @staticmethod
    def generate_react_native_config() -> Dict[str, Any]:
        """React Native 앱 설정 생성"""
        return {
            "app_name": "CherryAI Mobile",
            "bundle_id": "com.cherryai.mobile",
            "version": "1.0.0",
            "api_config": {
                "base_url": "https://api.cherryai.com",
                "timeout": 30000,
                "retry_attempts": 3
            },
            "features": {
                "offline_analysis": True,
                "voice_queries": True,
                "push_notifications": True,
                "biometric_auth": True,
                "file_upload": True
            },
            "ui_config": {
                "primary_color": "#2E86AB",
                "secondary_color": "#A23B72", 
                "accent_color": "#F18F01",
                "dark_mode": True
            },
            "permissions": [
                "camera",
                "microphone",
                "storage",
                "notifications",
                "biometric"
            ]
        }
    
    @staticmethod
    def generate_electron_config() -> Dict[str, Any]:
        """Electron 데스크톱 앱 설정 생성"""
        return {
            "app_name": "CherryAI Desktop",
            "version": "1.0.0",
            "window_config": {
                "width": 1200,
                "height": 800,
                "min_width": 800,
                "min_height": 600,
                "resizable": True,
                "show": False,
                "webPreferences": {
                    "nodeIntegration": False,
                    "contextIsolation": True,
                    "enableRemoteModule": False
                }
            },
            "features": {
                "auto_updater": True,
                "native_menus": True,
                "tray_icon": True,
                "system_notifications": True,
                "file_associations": [".csv", ".xlsx", ".json"]
            },
            "build_config": {
                "productName": "CherryAI",
                "appId": "com.cherryai.desktop",
                "directories": {
                    "output": "dist",
                    "resources": "resources"
                },
                "files": [
                    "build/**/*",
                    "node_modules/**/*",
                    "!node_modules/electron-builder/**/*"
                ]
            }
        }
    
    @staticmethod
    def generate_pwa_config() -> Dict[str, Any]:
        """PWA 설정 생성"""
        return {
            "name": "CherryAI",
            "short_name": "CherryAI",
            "description": "AI-powered data analysis platform",
            "start_url": "/",
            "display": "standalone",
            "theme_color": "#2E86AB",
            "background_color": "#FFFFFF",
            "orientation": "portrait",
            "icons": [
                {
                    "src": "/icons/icon-192x192.png",
                    "sizes": "192x192",
                    "type": "image/png"
                },
                {
                    "src": "/icons/icon-512x512.png",
                    "sizes": "512x512",
                    "type": "image/png"
                }
            ],
            "features": {
                "offline_support": True,
                "push_notifications": True,
                "background_sync": True,
                "install_prompt": True
            }
        }

# 전역 인스턴스
_mobile_api = None

def get_mobile_api() -> MobileOptimizedAPI:
    """모바일 API 싱글톤 인스턴스 반환"""
    global _mobile_api
    if _mobile_api is None:
        _mobile_api = MobileOptimizedAPI()
    return _mobile_api

async def test_mobile_integration():
    """모바일 통합 테스트"""
    print("🧪 Mobile Integration 테스트 시작")
    
    try:
        mobile_api = get_mobile_api()
        device_id = "test_device_001"
        
        # 기기 등록 테스트
        device_info = {
            "os": "iOS",
            "version": "15.0",
            "model": "iPhone 13",
            "app_version": "1.0.0"
        }
        
        result = await mobile_api.register_device(
            device_id=device_id,
            platform=PlatformType.REACT_NATIVE,
            device_info=device_info,
            user_id="user_001"
        )
        print(f"✅ 기기 등록: {result['status']}")
        
        # 오프라인 분석 테스트
        analysis_data = {
            "json_data": [
                {"name": "Alice", "age": 25, "salary": 50000},
                {"name": "Bob", "age": 30, "salary": 60000},
                {"name": "Charlie", "age": 35, "salary": 70000}
            ]
        }
        
        analysis_result = await mobile_api.submit_offline_analysis(device_id, analysis_data)
        print(f"✅ 오프라인 분석 요청: {analysis_result['status']}")
        
        if analysis_result['status'] == 'success':
            job_id = analysis_result['job_id']
            
            # 잠시 대기 후 작업 상태 확인
            await asyncio.sleep(2)
            
            jobs_result = await mobile_api.get_offline_jobs(device_id)
            print(f"✅ 작업 상태 조회: {jobs_result['status']}")
            
            if jobs_result['status'] == 'success':
                print(f"📊 총 작업 수: {len(jobs_result['jobs'])}")
        
        # 음성 쿼리 테스트
        voice_result = await mobile_api.process_voice_query(
            device_id, 
            base64.b64encode(b"dummy_audio_data").decode()
        )
        print(f"✅ 음성 쿼리 처리: {voice_result['status']}")
        
        # 대시보드 테스트
        dashboard_result = await mobile_api.get_mobile_dashboard(device_id)
        print(f"✅ 모바일 대시보드: {dashboard_result['status']}")
        
        if dashboard_result['status'] == 'success':
            dashboard = dashboard_result['dashboard']
            print(f"📱 성공률: {dashboard['success_rate']:.1f}%")
            print(f"📱 미읽은 알림: {dashboard['unread_notifications']}개")
        
        # 앱 설정 생성 테스트
        rn_config = AppConfigGenerator.generate_react_native_config()
        electron_config = AppConfigGenerator.generate_electron_config()
        pwa_config = AppConfigGenerator.generate_pwa_config()
        
        print(f"✅ React Native 설정: {rn_config['app_name']}")
        print(f"✅ Electron 설정: {electron_config['app_name']}")
        print(f"✅ PWA 설정: {pwa_config['name']}")
        
        print("✅ Mobile Integration 테스트 완료")
        return True
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(test_mobile_integration()) 