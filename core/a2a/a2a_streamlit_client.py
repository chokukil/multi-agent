"""
A2A Streamlit Client - 고급 에이전트 간 통신 클라이언트
Enhanced with proper agent mapping and plan execution handling
"""

import json
import httpx
from datetime import datetime
from typing import Dict, Any, List, AsyncGenerator
from core.utils.streamlit_context import safe_error, safe_warning, safe_success, safe_info, has_streamlit_context


class A2AStreamlitClient:
    """A2A 프로토콜을 사용한 Streamlit 클라이언트"""

    def __init__(self, agents_info: Dict, timeout: float = 180.0):
        """클라이언트 초기화"""
        self._agents_info = agents_info
        self._timeout = timeout
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(timeout))
        self._debug_log("A2A Streamlit 클라이언트 초기화 완료")

    def _debug_log(self, message: str, level: str = "info"):
        """디버깅 로그 출력"""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        
        if level == "error":
            log_msg = f"[{timestamp}] ❌ A2A ERROR: {message}"
        elif level == "warning":
            log_msg = f"[{timestamp}] ⚠️  A2A WARNING: {message}"
        elif level == "success":
            log_msg = f"[{timestamp}] ✅ A2A SUCCESS: {message}"
        else:
            log_msg = f"[{timestamp}] ℹ️  A2A DEBUG: {message}"
        
        print(log_msg)
        
        try:
            import os
            os.makedirs("logs", exist_ok=True)
            with open("logs/streamlit_debug.log", "a", encoding="utf-8") as f:
                f.write(f"{log_msg}\n")
                f.flush()
        except:
            pass
        
        # 안전한 Streamlit 호출
        if has_streamlit_context():
            if level == "error":
                safe_error(f"🐛 A2A DEBUG: {message}")
            elif level == "warning":
                safe_warning(f"🐛 A2A DEBUG: {message}")
            elif level == "success":
                safe_success(f"🐛 A2A DEBUG: {message}")
            else:
                safe_info(f"🐛 A2A DEBUG: {message}")

    async def get_plan(self, prompt: str, session_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """오케스트레이터에게 계획 요청 (세션 정보 및 파일 정보 포함)"""
        self._debug_log(f"🧠 오케스트레이터 계획 요청 시작: {prompt[:100]}...")
        
        orchestrator_url = "http://localhost:8100"
        message_id = f"plan_request_{int(datetime.now().timestamp())}"
        
        self._debug_log(f"📤 오케스트레이터 요청 URL: {orchestrator_url}")
        self._debug_log(f"📤 메시지 ID: {message_id}")
        
        # 메시지 parts 구성 - 텍스트 프롬프트 + 세션 정보
        message_parts = [{"kind": "text", "text": prompt}]
        
        # 세션 정보 추가
        if session_context:
            self._debug_log(f"📊 세션 컨텍스트 정보: {session_context}")
            
            # 업로드된 파일 정보 추가
            if "uploaded_file_info" in session_context:
                file_info = session_context["uploaded_file_info"]
                self._debug_log(f"📁 업로드된 파일 정보: {file_info}")
                
                # 파일 정보를 데이터 part로 추가
                message_parts.append({
                    "kind": "data",
                    "data": {
                        "type": "file_reference",
                        "file_path": file_info.get("file_path"),
                        "file_name": file_info.get("file_name"),
                        "session_id": file_info.get("session_id"),
                        "data_shape": file_info.get("data_shape"),
                        "data_info": file_info.get("data_info")
                    }
                })
            
            # 세션 메타데이터 추가
            if "session_metadata" in session_context:
                message_parts.append({
                    "kind": "data",
                    "data": {
                        "type": "session_metadata",
                        **session_context["session_metadata"]
                    }
                })
        
        payload = {
            "jsonrpc": "2.0",
            "method": "message/send",
            "params": {
                "message": {
                    "messageId": message_id,
                    "role": "user",
                    "parts": message_parts
                }
            },
            "id": message_id
        }
        
        self._debug_log(f"📤 요청 페이로드: {json.dumps(payload, ensure_ascii=False, indent=2)}")
        
        try:
            self._debug_log("🌐 HTTP 클라이언트로 요청 전송 중...")
            response = await self._client.post(orchestrator_url, json=payload)
            
            self._debug_log(f"📥 HTTP 응답 상태: {response.status_code}")
            self._debug_log(f"📥 응답 헤더: {dict(response.headers)}")
            
            response.raise_for_status()
            
            response_data = response.json()
            self._debug_log("📥 응답 JSON 파싱 성공")
            self._debug_log(f"📥 응답 최상위 키: {list(response_data.keys())}")
            
            if "result" in response_data:
                result = response_data["result"]
                self._debug_log(f"📊 'result' 필드 타입: {type(result)}")
                self._debug_log(f"📊 'result' 키들: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
                
                if "artifacts" in result:
                    artifacts = result["artifacts"]
                    self._debug_log(f"📦 'artifacts' 개수: {len(artifacts)}")
                    
                    for i, artifact in enumerate(artifacts):
                        self._debug_log(f"  📦 Artifact {i+1}: {type(artifact)}")
                        if isinstance(artifact, dict):
                            self._debug_log(f"    📦 Artifact {i+1} 키들: {list(artifact.keys())}")
                            self._debug_log(f"    📦 Artifact {i+1} 이름: {artifact.get('name', 'unnamed')}")
            
            return response_data
            
        except httpx.HTTPStatusError as e:
            self._debug_log(f"❌ HTTP 오류: {e.response.status_code}: {e.response.text}", "error")
            return {"error": f"HTTP 오류: {e.response.status_code}"}
            
        except httpx.ConnectError as e:
            self._debug_log(f"❌ 연결 실패: {e}", "error")
            return {"error": f"연결 실패: {e}"}
            
        except Exception as e:
            self._debug_log(f"💥 오케스트레이터 요청 중 예상치 못한 오류: {e}", "error")
            import traceback
            self._debug_log(f"🔍 스택 트레이스: {traceback.format_exc()}", "error")
            return {"error": f"예상치 못한 오류: {e}"}

    async def stream_task(self, agent_name: str, prompt: str, data_id: str = None) -> AsyncGenerator[Dict[str, Any], None]:
        """전문 에이전트에게 작업을 요청하고 스트리밍 응답을 반환합니다."""
        self._debug_log(f"🎯 stream_task 시작 - 에이전트: {agent_name}")
        
        # CherryAI v8 오케스트레이터 에이전트 매핑 처리
        mapped_agent_name = agent_name
        if agent_name == "🧠 CherryAI v8 Universal Orchestrator":
            mapped_agent_name = "Orchestrator"
            self._debug_log(f"🔄 v8 오케스트레이터 매핑: {agent_name} → {mapped_agent_name}")
        
        agent_info = self._agents_info.get(mapped_agent_name)
        if not agent_info:
            self._debug_log(f"❌ '{mapped_agent_name}' 에이전트 정보를 찾을 수 없음", "error")
            
            # v8 오케스트레이터의 경우 이미 완료된 분석 결과 반환
            if agent_name == "🧠 CherryAI v8 Universal Orchestrator":
                self._debug_log("🧠 v8 오케스트레이터 분석 이미 완료됨 - 결과 반환", "success")
                yield {
                    "type": "message",
                    "content": {"text": "🧠 CherryAI v8 Universal Intelligence 분석이 이미 완료되었습니다."},
                    "final": False
                }
                yield {
                    "type": "message", 
                    "content": {"text": "✅ 종합 분석 보고서가 생성되었습니다."},
                    "final": True
                }
                return
            
            # 다른 에이전트 매핑 시도
            agent_mapping = self._get_agent_mapping()
            for key, value in agent_mapping.items():
                if value == agent_name:
                    fallback_agent = key
                    if fallback_agent in self._agents_info:
                        mapped_agent_name = fallback_agent
                        agent_info = self._agents_info[fallback_agent]
                        self._debug_log(f"🔄 대체 에이전트 매핑: {agent_name} → {fallback_agent}")
                        break
            
            if not agent_info:
                raise ValueError(f"'{agent_name}' 에이전트 정보를 찾을 수 없습니다.")

        url = f"http://localhost:{agent_info['port']}"
        task_id = f"stream-task-{datetime.now().timestamp()}"
        
        # 메시지 parts 구성
        message_parts = [{"kind": "text", "text": prompt}]
        
        # 데이터 참조 정보 추가 (DataManager 기반)
        if data_id:
            try:
                from core.data_manager import DataManager
                data_manager = DataManager()
                
                # 데이터 정보 조회 (DataManager에서 직접 DataFrame 확인)
                df = data_manager.get_dataframe(data_id)
                if df is not None:
                    data_reference = {
                        "data_id": data_id,
                        "source": "file_upload",
                        "shape": [df.shape[0], df.shape[1]],
                        "memory_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 2),
                        "schema": {
                            "columns": df.columns.tolist(),
                            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()}
                        },
                        "location": f"shared_dataframes/{data_id}.pkl"
                    }
                    
                    # 데이터 참조를 별도 part로 추가
                    message_parts.append({
                        "kind": "data",
                        "data": {"data_reference": data_reference}
                    })
                    
                    self._debug_log(f"📊 데이터 참조 추가됨: {data_id} (형태: {df.shape})")
                else:
                    self._debug_log(f"⚠️ 데이터를 찾을 수 없음: {data_id}", "warning")
                    
            except Exception as e:
                self._debug_log(f"⚠️ 데이터 참조 추가 실패: {e}", "warning")

        payload = {
            "jsonrpc": "2.0", 
            "method": "message/send",
            "params": {
                "message": {
                    "messageId": task_id,
                    "role": "user",
                    "parts": message_parts
                }
            }, 
            "id": task_id
        }

        try:
            self._debug_log(f"🚀 '{mapped_agent_name}' 에이전트에게 작업 요청 전송 중...")
            
            response = await self._client.post(url, json=payload)
            self._debug_log(f"📥 '{mapped_agent_name}' 응답 수신 - HTTP Status: {response.status_code}")
            
            response.raise_for_status()
            response_data = response.json()
            
            # A2A 프로토콜 응답 처리 - 실제 구조에 맞게 수정
            if "result" in response_data:
                result = response_data["result"]
                
                # 스트리밍 메시지 처리 개선
                message_chunks = []
                
                # 1. 기본 메시지 처리 (status.message)
                if "status" in result and "message" in result["status"]:
                    status_msg = result["status"]["message"]
                    if "parts" in status_msg:
                        for part in status_msg["parts"]:
                            if part.get("kind") == "text":
                                text = part.get("text", "")
                                if text.strip():
                                    message_chunks.append(text)
                
                # 2. 히스토리 메시지들 처리
                if "history" in result:
                    for msg in result["history"]:
                        if msg.get("role") == "agent" and "parts" in msg:
                            for part in msg["parts"]:
                                if part.get("kind") == "text":
                                    text = part.get("text", "")
                                    if text.strip():
                                        message_chunks.append(text)
                
                # 3. 직접 메시지 구조 처리
                if "message" in result and "parts" in result["message"]:
                    for part in result["message"]["parts"]:
                        if part.get("kind") == "text":
                            text = part.get("text", "")
                            if text.strip():
                                message_chunks.append(text)
                
                # 4. 메시지 청크들을 스트리밍으로 전송
                for i, chunk_text in enumerate(message_chunks):
                    is_final = (i == len(message_chunks) - 1) and "artifacts" not in result
                    
                    # 청크를 더 작은 단위로 분할하여 스트리밍 효과 연출
                    words = chunk_text.split()
                    word_chunks = []
                    current_chunk = ""
                    
                    for word in words:
                        if len(current_chunk) + len(word) + 1 > 50:  # 50자 단위로 분할
                            if current_chunk:
                                word_chunks.append(current_chunk.strip())
                            current_chunk = word
                        else:
                            current_chunk += " " + word if current_chunk else word
                    
                    if current_chunk:
                        word_chunks.append(current_chunk.strip())
                    
                    # 단어 청크들을 스트리밍
                    for j, word_chunk in enumerate(word_chunks):
                        is_chunk_final = is_final and (j == len(word_chunks) - 1)
                        
                        yield {
                            "type": "message",
                            "content": {"text": word_chunk},
                            "final": is_chunk_final
                        }
                        
                        # 스트리밍 효과를 위한 지연
                        import asyncio
                        await asyncio.sleep(0.1)
                
                # 5. 아티팩트 스트리밍 (즉시 전송)
                if "artifacts" in result:
                    for artifact in result["artifacts"]:
                        if "parts" in artifact:
                            artifact_name = artifact.get("name", "artifact")
                            artifact_metadata = artifact.get("metadata", {})
                            
                            self._debug_log(f"📦 아티팩트 처리: {artifact_name}")
                            
                            for part in artifact["parts"]:
                                artifact_data = part.get("text", "")
                                
                                # Plotly 차트인 경우 특별 처리
                                if artifact_metadata.get("content_type") == "application/vnd.plotly.v1+json":
                                    self._debug_log("📊 Plotly 차트 아티팩트 감지")
                                    
                                    try:
                                        import json
                                        chart_data = json.loads(artifact_data) if isinstance(artifact_data, str) else artifact_data
                                        
                                        yield {
                                            "type": "artifact",
                                            "content": {
                                                "name": artifact_name,
                                                "data": chart_data,
                                                "contentType": "application/vnd.plotly.v1+json",
                                                "metadata": artifact_metadata
                                            },
                                            "final": False
                                        }
                                        
                                    except json.JSONDecodeError as e:
                                        self._debug_log(f"❌ Plotly 차트 JSON 파싱 실패: {e}", "error")
                                        yield {
                                            "type": "artifact",
                                            "content": {
                                                "name": artifact_name,
                                                "data": artifact_data,
                                                "contentType": "text/plain",
                                                "metadata": artifact_metadata
                                            },
                                            "final": False
                                        }
                                else:
                                    # 일반 아티팩트 처리
                                    yield {
                                        "type": "artifact",
                                        "content": {
                                            "name": artifact_name,
                                            "data": artifact_data,
                                            "contentType": artifact_metadata.get("content_type", "text/plain"),
                                            "metadata": artifact_metadata
                                        },
                                        "final": False
                                    }
                
                # 6. 최종 완료 신호
                yield {
                    "type": "message",
                    "content": {"text": f"✅ {mapped_agent_name} 작업 완료"},
                    "final": True
                }
                                
            elif "error" in response_data:
                error_msg = response_data['error']['message']
                self._debug_log(f"❌ '{mapped_agent_name}' 오류: {error_msg}", "error")
                yield {
                    "type": "message", 
                    "content": {"text": f"❌ 오류: {error_msg}"},
                    "final": True
                }
            else:
                # 응답이 없는 경우
                yield {
                    "type": "message",
                    "content": {"text": f"⚠️ {mapped_agent_name}에서 응답이 없습니다."},
                    "final": True
                }
                
        except Exception as e:
            self._debug_log(f"❌ '{mapped_agent_name}' 오류: {type(e).__name__}: {e}", "error")
            yield {
                "type": "message",
                "content": {"text": f"❌ 연결 오류: {e}"},
                "final": True
            }

    async def close(self):
        """클라이언트 연결을 종료합니다."""
        await self._client.aclose()
        self._debug_log("A2A Streamlit 클라이언트 연결 종료")

    def parse_orchestration_plan(self, orchestrator_response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """A2A 표준 기반 오케스트레이터 응답 파싱"""
        self._debug_log("🔍 A2A 표준 기반 계획 파싱 시작...")
        self._debug_log(f"📋 응답 타입: {type(orchestrator_response)}")
        
        if not isinstance(orchestrator_response, dict):
            self._debug_log("❌ 응답이 딕셔너리가 아님", "error")
            return []
        
        self._debug_log(f"📋 응답 최상위 키들: {list(orchestrator_response.keys())}")
        
        try:
            # A2A 표준 JSON-RPC 2.0 응답 구조 확인
            if "result" in orchestrator_response:
                result = orchestrator_response["result"]
                self._debug_log(f"📊 A2A result 타입: {type(result)}")
                
                if isinstance(result, dict):
                    # A2A 표준: history와 status 구조
                    if "history" in result and "status" in result:
                        return self._parse_a2a_standard_response(result)
                    
                    # 직접 메시지 구조
                    elif "message" in result:
                        return self._parse_direct_message_response(result)
                    
                    # 기타 직접 응답 구조
                    else:
                        return self._parse_direct_response(result)
            
            # 비표준 직접 응답 (폴백)
            else:
                return self._parse_direct_response(orchestrator_response)
                
        except Exception as e:
            self._debug_log(f"❌ 계획 파싱 오류: {type(e).__name__}: {e}", "error")
            import traceback
            self._debug_log(f"🔍 스택 트레이스: {traceback.format_exc()}", "error")
            return []

    def _parse_a2a_standard_response(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """A2A 표준 응답 구조 파싱 (artifacts + history + status)"""
        self._debug_log("🎯 A2A 표준 응답 구조 파싱 중...")
        
        try:
            # 🎯 우선순위 1: CherryAI v8 comprehensive_analysis 아티팩트 처리
            if "artifacts" in result:
                artifacts = result["artifacts"]
                self._debug_log(f"📦 {len(artifacts)}개 아티팩트 발견")
                
                for artifact in artifacts:
                    artifact_name = artifact.get("name", "")
                    
                    # CherryAI v8 오케스트레이터: comprehensive_analysis 아티팩트 처리 (최우선)
                    if artifact_name == "comprehensive_analysis":
                        self._debug_log(f"🧠 CherryAI v8 종합 분석 아티팩트 발견: {artifact_name}")
                        parts = artifact.get("parts", [])
                        self._debug_log(f"🔍 v8 아티팩트 parts 개수: {len(parts)}")
                        
                        for i, part in enumerate(parts):
                            part_kind = part.get("kind", "unknown")
                            self._debug_log(f"🔍 Part {i+1}: kind={part_kind}")
                            
                            if part_kind == "text":
                                analysis_text = part.get("text", "")
                                self._debug_log(f"📝 v8 텍스트 길이: {len(analysis_text)}")
                                
                                if analysis_text:
                                    self._debug_log(f"📝 v8 종합 분석 결과 발견: {len(analysis_text)} chars")
                                    # v8 오케스트레이터는 최종 분석 결과를 제공하므로 단일 단계로 처리
                                    v8_step = {
                                        "step_number": 1,
                                        "agent_name": "🧠 CherryAI v8 Universal Orchestrator",
                                        "task_description": "종합 분석 및 최종 보고서 생성",
                                        "reasoning": "CherryAI v8 Universal Intelligent Orchestrator의 종합 분석 결과",
                                        "expected_result": "완료된 종합 분석 보고서",
                                        "final_analysis": analysis_text,  # 실제 분석 결과 포함
                                        "parameters": {
                                            "user_instructions": "CherryAI v8 Universal Intelligence 종합 분석",
                                            "priority": "high",
                                            "analysis_complete": True  # 분석 완료 플래그
                                        }
                                    }
                                    self._debug_log(f"✅ v8 단계 생성 완료: {v8_step['agent_name']}")
                                    return [v8_step]
                                else:
                                    self._debug_log("❌ v8 분석 텍스트가 비어있음", "warning")
                            else:
                                self._debug_log(f"⚠️ v8 Part {i+1}이 텍스트가 아님: {part_kind}", "warning")
                        
                        self._debug_log("❌ v8 아티팩트에서 유효한 텍스트를 찾을 수 없음", "error")
                        # v8 아티팩트가 있지만 텍스트가 없는 경우에도 즉시 반환하여 history 처리 방지
                        return []
            
            # 🎯 우선순위 2: 기존 execution_plan 아티팩트 처리
            if "artifacts" in result:
                artifacts = result["artifacts"]
                
                for artifact in artifacts:
                    artifact_name = artifact.get("name", "")
                    
                    # 실행 계획 아티팩트 확인 (기존 로직 유지)
                    if artifact_name in ["execution_plan", "execution_plan.json"] or "execution_plan" in artifact_name:
                        metadata = artifact.get("metadata", {})
                        self._debug_log(f"📋 실행 계획 아티팩트 발견: {artifact_name}")
                        
                        # 메타데이터 확인 (선택적)
                        if metadata.get("plan_type") == "ai_ds_team_orchestration" or metadata.get("content_type") == "application/json":
                            parts = artifact.get("parts", [])
                            for part in parts:
                                # TextPart with JSON data
                                if part.get("kind") == "text":
                                    plan_text = part.get("text", "")
                                    if plan_text:
                                        self._debug_log(f"📝 아티팩트에서 JSON 계획 발견: {len(plan_text)} chars")
                                        return self._extract_plan_from_artifact_text(plan_text)
                                # DataPart with direct JSON
                                elif "data" in part:
                                    plan_data = part.get("data")
                                    if isinstance(plan_data, dict):
                                        self._debug_log("📊 아티팩트에서 직접 JSON 데이터 발견")
                                        return self._process_artifact_plan_data(plan_data)
                        else:
                            # 메타데이터가 없어도 이름으로 판단하여 파싱 시도
                            parts = artifact.get("parts", [])
                            for part in parts:
                                if part.get("kind") == "text":
                                    plan_text = part.get("text", "")
                                    if plan_text:
                                        self._debug_log(f"📝 메타데이터 없이 아티팩트에서 계획 텍스트 발견: {len(plan_text)} chars")
                                        return self._extract_plan_from_artifact_text(plan_text)
            
            # 🎯 우선순위 3: history에서 agent 메시지 찾기 (아티팩트가 없을 때만)
            history = result.get("history", [])
            
            # 기존 history 파싱 로직 (폴백)
            for entry in history:
                if entry.get("role") == "agent" and "message" in entry:
                    message = entry["message"]
                    if "parts" in message:
                        for part in message["parts"]:
                            if part.get("kind") == "text":
                                plan_text = part.get("text", "")
                                if plan_text:
                                    self._debug_log(f"📝 History에서 계획 텍스트 발견: {len(plan_text)} chars")
                                    return self._extract_plan_from_text(plan_text)
            
            # 🎯 우선순위 4: status.message에서 확인 (최후 수단)
            status = result.get("status", {})
            if "message" in status:
                message = status["message"]
                if "parts" in message:
                    for part in message["parts"]:
                        if part.get("kind") == "text":
                            plan_text = part.get("text", "")
                            if plan_text:
                                self._debug_log(f"📝 Status에서 계획 텍스트 발견: {len(plan_text)} chars")
                                return self._extract_plan_from_text(plan_text)
            
            self._debug_log("⚠️ A2A 표준 응답에서 계획을 찾을 수 없음", "warning")
            return []
            
        except Exception as e:
            self._debug_log(f"❌ A2A 표준 응답 파싱 실패: {e}", "error")
            return []

    def _extract_plan_from_artifact_text(self, text: str) -> List[Dict[str, Any]]:
        """아티팩트 텍스트에서 JSON 계획 추출"""
        self._debug_log(f"📝 아티팩트에서 계획 추출 중... (길이: {len(text)})")
        
        try:
            # 직접 JSON 파싱 시도 (아티팩트는 이미 정제된 JSON)
            plan_data = json.loads(text)
            self._debug_log(f"📊 아티팩트 JSON 파싱 성공: {list(plan_data.keys())}")
            return self._process_artifact_plan_data(plan_data)
                
        except json.JSONDecodeError as e:
            self._debug_log(f"❌ 아티팩트 JSON 파싱 실패: {e}", "error")
            # 폴백: 기존 텍스트 추출 방법 사용
            return self._extract_plan_from_text(text)
        except Exception as e:
            self._debug_log(f"❌ 아티팩트 계획 추출 실패: {e}", "error")
            return []

    def _process_artifact_plan_data(self, plan_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """아티팩트 계획 데이터 처리"""
        self._debug_log(f"🔄 아티팩트 계획 데이터 처리 중... 키들: {list(plan_data.keys())}")
        
        try:
            # plan_executed 형식 (오케스트레이터 v6 표준)
            if "plan_executed" in plan_data:
                steps = plan_data["plan_executed"]
                self._debug_log(f"✅ 'plan_executed' 형식으로 {len(steps)}개 단계 발견")
                return self._process_steps(steps)
            
            # steps 형식
            elif "steps" in plan_data:
                steps = plan_data["steps"]
                self._debug_log(f"✅ 'steps' 형식으로 {len(steps)}개 단계 발견")
                return self._process_steps(steps)
            
            # A2A 표준 오케스트레이션 계획
            elif plan_data.get("plan_type") == "ai_ds_team_orchestration":
                steps = plan_data.get("steps", [])
                if steps:
                    self._debug_log(f"✅ A2A 표준 오케스트레이션 계획: {len(steps)}개 단계")
                    return self._process_steps(steps)
            
            # 리스트 형식
            elif isinstance(plan_data, list):
                self._debug_log(f"✅ 리스트 형식으로 {len(plan_data)}개 단계 발견")
                return self._process_steps(plan_data)
            
            else:
                self._debug_log(f"❌ 알 수 없는 아티팩트 계획 형식: {list(plan_data.keys())}", "warning")
                return []
                
        except Exception as e:
            self._debug_log(f"❌ 아티팩트 계획 데이터 처리 실패: {e}", "error")
            return []

    def _parse_direct_message_response(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """직접 메시지 응답 구조 파싱"""
        self._debug_log("🎯 직접 메시지 응답 파싱 중...")
        
        try:
            message = result.get("message", {})
            if "parts" in message:
                for part in message["parts"]:
                    if part.get("kind") == "text":
                        plan_text = part.get("text", "")
                        if plan_text:
                            self._debug_log(f"📝 직접 메시지에서 계획 텍스트 발견: {len(plan_text)} chars")
                            return self._extract_plan_from_text(plan_text)
            
            self._debug_log("⚠️ 직접 메시지에서 계획을 찾을 수 없음", "warning")
            return []
            
        except Exception as e:
            self._debug_log(f"❌ 직접 메시지 응답 파싱 실패: {e}", "error")
            return []

    def _parse_direct_response(self, response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """직접 응답 구조 파싱 (steps, plan_executed 등)"""
        self._debug_log("🎯 직접 응답 구조 파싱 중...")
        
        try:
            # steps 형식
            if "steps" in response:
                steps = response["steps"]
                self._debug_log(f"✅ 'steps' 형식으로 {len(steps)}개 단계 발견")
                return self._process_steps(steps)
            
            # plan_executed 형식
            elif "plan_executed" in response:
                steps = response["plan_executed"]
                self._debug_log(f"✅ 'plan_executed' 형식으로 {len(steps)}개 단계 발견")
                return self._process_steps(steps)
            
            # history 내부 확인 (A2A 표준 구조의 일부)
            elif "history" in response:
                history = response["history"]
                for entry in history:
                    if entry.get("role") == "agent" and "message" in entry:
                        message = entry["message"]
                        if "parts" in message:
                            for part in message["parts"]:
                                if part.get("kind") == "text":
                                    plan_text = part.get("text", "")
                                    if plan_text:
                                        self._debug_log(f"📝 History에서 계획 텍스트 발견: {len(plan_text)} chars")
                                        return self._extract_plan_from_text(plan_text)
            
            # status 내부 확인 (중첩 구조)
            elif "status" in response:
                status = response["status"]
                if isinstance(status, dict) and "message" in status:
                    message = status["message"]
                    if "parts" in message:
                        for part in message["parts"]:
                            if part.get("kind") == "text":
                                plan_text = part.get("text", "")
                                if plan_text:
                                    self._debug_log(f"📝 Status 메시지에서 계획 텍스트 발견: {len(plan_text)} chars")
                                    return self._extract_plan_from_text(plan_text)
            
            # 리스트 형식
            elif isinstance(response, list):
                self._debug_log(f"✅ 리스트 형식으로 {len(response)}개 단계 발견")
                return self._process_steps(response)
            
            # 문자열 형식 (JSON 파싱 시도)
            elif isinstance(response, str):
                self._debug_log("🔍 문자열 응답 감지, JSON 파싱 시도")
                try:
                    plan_data = json.loads(response)
                    return self._parse_direct_response(plan_data)
                except json.JSONDecodeError:
                    self._debug_log("❌ 문자열 응답이 유효한 JSON이 아님", "warning")
                    return []
            
            self._debug_log("⚠️ 직접 응답에서 계획을 찾을 수 없음", "warning")
            return []
            
        except Exception as e:
            self._debug_log(f"❌ 직접 응답 파싱 실패: {e}", "error")
            return []

    def _extract_plan_from_text(self, text: str) -> List[Dict[str, Any]]:
        """텍스트에서 JSON 계획 추출"""
        self._debug_log(f"📝 텍스트에서 계획 추출 중... (길이: {len(text)})")
        
        try:
            # JSON 블록 찾기 (```json ... ``` 형식)
            import re
            json_matches = re.findall(r'```json\s*(.*?)\s*```', text, re.DOTALL)
            if json_matches:
                plan_text = json_matches[0].strip()
                self._debug_log("✅ JSON 블록에서 계획 발견")
            else:
                # 직접 JSON 파싱 시도
                plan_text = text.strip()
                self._debug_log("🔍 직접 JSON 파싱 시도")
            
            plan_data = json.loads(plan_text)
            self._debug_log(f"📊 파싱된 계획 키들: {list(plan_data.keys())}")
            
            # 다양한 형식 지원
            if "steps" in plan_data:
                return self._process_steps(plan_data["steps"])
            elif "plan_executed" in plan_data:
                return self._process_steps(plan_data["plan_executed"])
            elif isinstance(plan_data, list):
                return self._process_steps(plan_data)
            else:
                self._debug_log(f"❌ 알 수 없는 계획 형식: {list(plan_data.keys())}", "warning")
                return []
                
        except json.JSONDecodeError as e:
            self._debug_log(f"❌ JSON 파싱 실패: {e}", "error")
            self._debug_log(f"🔍 파싱 시도한 텍스트: {text[:200]}...", "error")
            return []
        except Exception as e:
            self._debug_log(f"❌ 텍스트 계획 추출 실패: {e}", "error")
            return []

    def _process_steps(self, steps: List[Dict]) -> List[Dict[str, Any]]:
        """단계 리스트를 표준화된 형식으로 처리"""
        self._debug_log(f"🔄 {len(steps)}개 단계 처리 중...")
        
        if not isinstance(steps, list):
            self._debug_log("❌ 단계가 리스트가 아님", "error")
            return []
        
        valid_steps = []
        agent_mapping = self._get_agent_mapping()
        
        for i, step in enumerate(steps):
            if not isinstance(step, dict):
                self._debug_log(f"⚠️ 단계 {i+1}이 딕셔너리가 아님", "warning")
                continue
            
            # 에이전트 이름 추출 및 매핑
            agent_name = step.get("agent_name") or step.get("agent", "unknown")
            mapped_agent = agent_mapping.get(agent_name, agent_name)
            
            # 에이전트별 구체적인 예상 결과 생성
            expected_result = self._generate_expected_result(mapped_agent, step.get("task_description", ""))
            
            # 표준화된 단계 생성
            standardized_step = {
                "step_number": step.get("step_number", step.get("step", i + 1)),
                "agent_name": mapped_agent,
                "task_description": step.get("task_description") or step.get("description") or step.get("task", ""),
                "reasoning": step.get("reasoning", f"{mapped_agent} 전문 역량 활용"),
                "expected_result": expected_result,
                "parameters": {
                    "user_instructions": step.get("task_description") or step.get("description") or step.get("task", ""),
                    "priority": step.get("priority", "medium")
                }
            }
            
            # 에이전트 사용 가능성 확인
            if mapped_agent in self._agents_info:
                valid_steps.append(standardized_step)
                self._debug_log(f"  ✅ 단계 {standardized_step['step_number']}: {mapped_agent}")
            else:
                self._debug_log(f"  ⚠️ 에이전트 '{mapped_agent}' 사용 불가능", "warning")
                # 사용 가능한 에이전트로 대체
                available_agents = [name for name in self._agents_info.keys() if name != "Orchestrator"]
                if available_agents:
                    fallback_agent = available_agents[0]
                    standardized_step["agent_name"] = fallback_agent
                    standardized_step["task_description"] += f" (원래: {mapped_agent})"
                    standardized_step["expected_result"] = self._generate_expected_result(fallback_agent, standardized_step["task_description"])
                    valid_steps.append(standardized_step)
                    self._debug_log(f"  🔄 대체 에이전트 사용: {fallback_agent}")
        
        self._debug_log(f"🎉 총 {len(valid_steps)}개 유효한 단계 처리 완료")
        return valid_steps

    def _generate_expected_result(self, agent_name: str, task_description: str) -> str:
        """에이전트별 구체적인 예상 결과 생성"""
        
        # 에이전트별 전문 예상 결과
        if agent_name == "📁 Data Loader":
            return "로드된 데이터셋 정보, 컬럼 구조, 데이터 타입 요약 및 기본 품질 검증 결과"
        
        elif agent_name == "🧹 Data Cleaning":
            return "결측값 처리 보고서, 중복 데이터 제거 현황, 데이터 타입 최적화 결과 및 정제된 데이터셋"
        
        elif agent_name == "🔍 EDA Tools":
            return "기초 통계량, 분포 분석, 상관관계 매트릭스, 이상값 탐지 결과 및 데이터 패턴 인사이트"
        
        elif agent_name == "📊 Data Visualization":
            return "히스토그램, 산점도, 박스플롯, 히트맵 등 시각화 차트 및 패턴 해석 보고서"
        
        elif agent_name == "🔧 Data Wrangling":
            return "데이터 변환 스크립트, 새로운 파생 변수, 데이터 구조 재편성 결과 및 처리 로그"
        
        elif agent_name == "⚙️ Feature Engineering":
            return "새로운 특성 변수, 특성 중요도 분석, 차원 축소 결과 및 특성 선택 추천사항"
        
        elif agent_name == "🗄️ SQL Database":
            return "SQL 쿼리 결과, 데이터베이스 스키마 분석, 조인 테이블 정보 및 성능 최적화 제안"
        
        elif agent_name == "🤖 H2O ML":
            return "AutoML 모델 성능 비교, 최적 모델 추천, 예측 정확도 지표 및 모델 해석 결과"
        
        elif agent_name == "📈 MLflow Tools":
            return "실험 추적 결과, 모델 버전 관리 정보, 성능 메트릭 비교 및 모델 배포 가이드"
        
        else:
            # 기본 예상 결과 (알 수 없는 에이전트)
            return "전문 분석 결과 및 도메인별 인사이트"

    def _get_agent_mapping(self) -> Dict[str, str]:
        """에이전트 이름 매핑 테이블 반환"""
        return {
            # 기본 에이전트 이름
            "data_loader": "📁 Data Loader",
            "data_cleaning": "🧹 Data Cleaning", 
            "eda_tools": "🔍 EDA Tools",
            "data_visualization": "📊 Data Visualization",
            "data_wrangling": "🔧 Data Wrangling",
            "feature_engineering": "⚙️ Feature Engineering",
            "sql_database": "🗄️ SQL Database",
            "h2o_ml": "🤖 H2O ML",
            "mlflow_tools": "📈 MLflow Tools",
            
            # 오케스트레이터 응답에서 사용하는 전체 이름
            "AI_DS_Team DataLoaderToolsAgent": "📁 Data Loader",
            "AI_DS_Team DataCleaningAgent": "🧹 Data Cleaning",
            "AI_DS_Team EDAToolsAgent": "🔍 EDA Tools", 
            "AI_DS_Team DataVisualizationAgent": "📊 Data Visualization",
            "AI_DS_Team DataWranglingAgent": "🔧 Data Wrangling",
            "AI_DS_Team FeatureEngineeringAgent": "⚙️ Feature Engineering",
            "AI_DS_Team SQLDatabaseAgent": "🗄️ SQL Database",
            "AI_DS_Team H2OMLAgent": "🤖 H2O ML",
            "AI_DS_Team MLflowAgent": "📈 MLflow Tools",
            
            # 다양한 변형 이름들
            "DataLoaderToolsAgent": "📁 Data Loader",
            "DataCleaningAgent": "🧹 Data Cleaning",
            "EDAToolsAgent": "🔍 EDA Tools",
            "DataVisualizationAgent": "📊 Data Visualization",
            "DataWranglingAgent": "🔧 Data Wrangling",
            "FeatureEngineeringAgent": "⚙️ Feature Engineering",
            "SQLDatabaseAgent": "🗄️ SQL Database",
            "H2OMLAgent": "🤖 H2O ML",
            "MLflowAgent": "📈 MLflow Tools"
        }
