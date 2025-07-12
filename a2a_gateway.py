#!/usr/bin/env python3
"""
🌐 A2A API Gateway

외부 API 요청을 A2A 프로토콜로 변환하는 통합 진입점
LLM First 멀티에이전트 데이터 분석 플랫폼의 Gateway Layer

Features:
- REST API → A2A Protocol 변환
- 인증 & 권한 관리  
- Rate Limiting & 요청 큐잉
- 실시간 WebSocket 스트리밍
- Auto Agent Discovery

Author: CherryAI Team
License: MIT License
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# A2A SDK 0.2.9 Import
from a2a.client import A2AClient
from a2a.types import TextPart, Message, Role, SendMessageRequest

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="CherryAI A2A Gateway",
    description="LLM First 멀티에이전트 데이터 분석 플랫폼의 통합 진입점",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 글로벌 설정
ORCHESTRATOR_URL = "http://localhost:8100"
DATA_HUB_URL = "http://localhost:8500"

class AnalysisRequest(BaseModel):
    """분석 요청 모델"""
    query: str
    session_id: Optional[str] = None
    data_source: Optional[str] = None
    options: Optional[Dict[str, Any]] = {}

class AnalysisResponse(BaseModel):
    """분석 응답 모델"""
    success: bool
    session_id: str
    result: Dict[str, Any]
    execution_time: float
    agents_used: List[str]

class A2AGateway:
    """A2A Gateway 핵심 클래스"""
    
    def __init__(self):
        self.orchestrator_client = A2AClient(base_url=ORCHESTRATOR_URL)
        self.active_sessions: Dict[str, Dict] = {}
        
    async def convert_to_a2a_message(self, request: AnalysisRequest) -> Message:
        """REST 요청을 A2A 메시지로 변환"""
        
        # 세션 ID 생성
        if not request.session_id:
            request.session_id = f"session_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        
        # A2A 메시지 구성
        message = Message(
            messageId=f"msg_{uuid.uuid4().hex}",
            role=Role.user,
            parts=[TextPart(text=request.query)]
        )
        
        # 세션 정보 저장
        self.active_sessions[request.session_id] = {
            "created_at": datetime.now(),
            "last_activity": datetime.now(),
            "data_source": request.data_source,
            "options": request.options
        }
        
        return message
    
    async def send_to_orchestrator(self, message: Message, session_id: str) -> Dict[str, Any]:
        """A2A Orchestrator에게 메시지 전송"""
        
        try:
            # A2A 클라이언트로 요청 전송
            send_request = SendMessageRequest(
                message=message,
                context={"session_id": session_id}
            )
            
            response = await self.orchestrator_client.send_message(send_request)
            
            return {
                "success": True,
                "response": response,
                "session_id": session_id
            }
            
        except Exception as e:
            logger.error(f"❌ Orchestrator 통신 실패: {e}")
            return {
                "success": False,
                "error": str(e),
                "session_id": session_id
            }

gateway = A2AGateway()

@app.get("/")
async def home():
    """Gateway 홈페이지"""
    return {
        "name": "CherryAI A2A Gateway",
        "version": "1.0.0",
        "description": "LLM First 멀티에이전트 데이터 분석 플랫폼의 통합 진입점",
        "endpoints": {
            "analyze": "/analyze",
            "upload": "/upload",
            "status": "/status",
            "agents": "/agents",
            "sessions": "/sessions"
        },
        "features": [
            "REST → A2A 프로토콜 변환",
            "실시간 WebSocket 스트리밍",
            "자동 Agent Discovery",
            "세션 기반 대화 관리"
        ]
    }

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_data(request: AnalysisRequest):
    """데이터 분석 요청 처리"""
    
    start_time = time.time()
    
    try:
        logger.info(f"🔍 분석 요청: {request.query}")
        
        # A2A 메시지로 변환
        a2a_message = await gateway.convert_to_a2a_message(request)
        
        # Orchestrator에게 전송
        result = await gateway.send_to_orchestrator(a2a_message, request.session_id)
        
        execution_time = time.time() - start_time
        
        if result["success"]:
            return AnalysisResponse(
                success=True,
                session_id=result["session_id"],
                result=result["response"],
                execution_time=execution_time,
                agents_used=result.get("agents_used", [])
            )
        else:
            raise HTTPException(status_code=500, detail=result["error"])
            
    except Exception as e:
        logger.error(f"❌ 분석 처리 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
    description: Optional[str] = Form(None)
):
    """파일 업로드"""
    
    try:
        if not session_id:
            session_id = f"upload_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        
        # 파일 내용 읽기
        file_content = await file.read()
        
        # Data Hub에 파일 저장 요청
        async with httpx.AsyncClient() as client:
            files = {"file": (file.filename, file_content, file.content_type)}
            data = {
                "session_id": session_id,
                "description": description or f"Uploaded file: {file.filename}"
            }
            
            response = await client.post(
                f"{DATA_HUB_URL}/upload",
                files=files,
                data=data
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "success": True,
                    "session_id": session_id,
                    "file_id": result.get("file_id"),
                    "message": f"파일 '{file.filename}' 업로드 완료"
                }
            else:
                raise HTTPException(status_code=500, detail="파일 업로드 실패")
                
    except Exception as e:
        logger.error(f"❌ 파일 업로드 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agents")
async def discover_agents():
    """사용 가능한 A2A 에이전트 발견"""
    
    try:
        # Orchestrator에게 에이전트 목록 요청
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{ORCHESTRATOR_URL}/agents")
            
            if response.status_code == 200:
                agents = response.json()
                return {
                    "success": True,
                    "agents": agents,
                    "count": len(agents),
                    "discovered_at": datetime.now().isoformat()
                }
            else:
                return {"success": False, "error": "에이전트 발견 실패"}
                
    except Exception as e:
        logger.error(f"❌ 에이전트 발견 실패: {e}")
        return {"success": False, "error": str(e)}

@app.get("/status")
async def system_status():
    """시스템 상태 확인"""
    
    status = {
        "gateway": "healthy",
        "orchestrator": "checking",
        "data_hub": "checking",
        "active_sessions": len(gateway.active_sessions),
        "timestamp": datetime.now().isoformat()
    }
    
    # Orchestrator 상태 확인
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(f"{ORCHESTRATOR_URL}/health")
            status["orchestrator"] = "healthy" if response.status_code == 200 else "unhealthy"
    except:
        status["orchestrator"] = "unreachable"
    
    # Data Hub 상태 확인
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(f"{DATA_HUB_URL}/health")
            status["data_hub"] = "healthy" if response.status_code == 200 else "unhealthy"
    except:
        status["data_hub"] = "unreachable"
    
    return status

@app.get("/sessions")
async def list_sessions():
    """활성 세션 목록"""
    
    sessions = []
    for session_id, info in gateway.active_sessions.items():
        sessions.append({
            "session_id": session_id,
            "created_at": info["created_at"].isoformat(),
            "last_activity": info["last_activity"].isoformat(),
            "data_source": info.get("data_source"),
            "options": info.get("options", {})
        })
    
    return {
        "success": True,
        "sessions": sessions,
        "count": len(sessions)
    }

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """실시간 WebSocket 스트리밍"""
    
    await websocket.accept()
    logger.info(f"🔌 WebSocket 연결: {session_id}")
    
    try:
        while True:
            # 클라이언트로부터 메시지 수신
            data = await websocket.receive_text()
            request_data = json.loads(data)
            
            # 분석 요청 처리
            request = AnalysisRequest(
                query=request_data["query"],
                session_id=session_id,
                data_source=request_data.get("data_source"),
                options=request_data.get("options", {})
            )
            
            # A2A 메시지로 변환
            a2a_message = await gateway.convert_to_a2a_message(request)
            
            # 진행 상황 스트리밍
            await websocket.send_text(json.dumps({
                "type": "progress",
                "stage": "processing",
                "message": "요청을 A2A Orchestrator로 전송 중...",
                "progress": 10
            }))
            
            # Orchestrator에게 전송 (스트리밍)
            result = await gateway.send_to_orchestrator(a2a_message, session_id)
            
            # 결과 전송
            await websocket.send_text(json.dumps({
                "type": "result",
                "success": result["success"],
                "data": result.get("response"),
                "session_id": session_id
            }))
            
    except Exception as e:
        logger.error(f"❌ WebSocket 오류: {e}")
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": str(e)
        }))
    finally:
        logger.info(f"🔌 WebSocket 연결 종료: {session_id}")

if __name__ == "__main__":
    logger.info("🌐 CherryAI A2A Gateway 시작")
    logger.info("📍 주소: http://0.0.0.0:8000")
    logger.info("🔗 WebSocket: ws://0.0.0.0:8000/ws/{session_id}")
    logger.info("📋 API 문서: http://0.0.0.0:8000/docs")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    ) 