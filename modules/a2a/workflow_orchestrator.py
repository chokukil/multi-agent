import os, tempfile, pandas as pd, asyncio, logging
from typing import Any, Dict, List, Optional, AsyncGenerator
from modules.a2a.agent_client import A2AAgentClient
from modules.artifacts.text_to_artifact_generator import TextToArtifactGenerator

logger = logging.getLogger(__name__)

EDA_ENDPOINTS = [
    os.environ.get("A2A_EDA_URL", "http://localhost:8312"),
    os.environ.get("A2A_PANDAS_URL", "http://localhost:8315"),
]

class A2AWorkflowOrchestrator:
    """A2A SDK 0.2.9 표준 준수 워크플로우 오케스트레이터 (SSE 스트리밍 지원)"""
    
    def __init__(self):
        self.chunk_delay = 0.01  # 0.01초 자연스러운 딜레이
        self.artifact_generator = TextToArtifactGenerator()  # 아티팩트 생성기
        
    def _materialize_dataset(self, df: pd.DataFrame) -> str:
        """데이터프레임을 임시 CSV 파일로 저장"""
        tmp = tempfile.NamedTemporaryFile(prefix="cherry_", suffix=".csv", delete=False)
        df.to_csv(tmp.name, index=False)
        return tmp.name

    def execute_workflow(self, selected_agents: List[Dict[str,Any]], query: str,
                         data: Optional[Dict[str,Any]] = None, meta: Optional[Dict[str,Any]] = None) -> Dict[str,Any]:
        """동기 워크플로우 실행 (기존 호환성 유지)"""
        datasets = (data or {}).get("datasets", {})
        files: List[str] = []
        for _, df in datasets.items():
            if isinstance(df, pd.DataFrame):
                files.append(self._materialize_dataset(df))

        # 사용자 친화적 상태 메시지
        status_message = f"📊 **분석 진행 상황**\n\n✅ {len(files)}개 데이터셋 처리 완료\n🔄 에이전트 분석 중...\n\n"
        user_text = f"[TASK=EDA]\nquery={query}"
        last_err = None
        
        for ep in EDA_ENDPOINTS:
            client = A2AAgentClient(ep, timeout=60)
            
            # 1) 헬스 체크
            if not client.health_check():
                logger.warning(f"에이전트 {ep} 헬스 체크 실패")
                continue
            
            # 2) 스트리밍 우선 시도 (스트리밍 미지원 에이전트 자동 감지)
            try:
                chunks = []
                streaming_supported = True
                
                for ch in client.stream_message(user_text, file_paths=files, meta={"task":"eda","files":files}):
                    # 스트리밍 미지원 감지
                    if '"code":-32004' in ch and 'Streaming is not supported' in ch:
                        logger.info(f"에이전트 {ep} 스트리밍 미지원 - 일반 메시지로 폴백")
                        streaming_supported = False
                        break
                    
                    clean_chunk = self._clean_a2a_response(ch)
                    if clean_chunk:
                        chunks.append(clean_chunk)
                
                if streaming_supported and chunks:
                    clean_result = "".join(chunks)
                    
                    # 아티팩트 생성
                    artifacts = self._generate_artifacts_from_response(
                        clean_result, datasets, "streaming_agent", query
                    )
                    
                    return {
                        "text": status_message + clean_result,
                        "artifacts": artifacts
                    }
                    
            except Exception as e:
                logger.warning(f"스트리밍 실패 {ep}: {e}")
                last_err = e
                
            # 3) 일반 메시지 폴백
            try:
                resp = client.send_message(user_text, file_paths=files, meta={"task":"eda","files":files}, dry_run=False)
                if "result" in resp:
                    raw_text = resp["result"].get("text") or resp["result"].get("message") or "EDA 결과가 수신되었습니다."
                    clean_text = self._clean_a2a_response(raw_text)
                    
                    # A2A 응답에서 실제 텍스트 추출
                    actual_response = self._extract_actual_response(resp)
                    
                    # 아티팩트 생성
                    artifacts = self._generate_artifacts_from_response(
                        actual_response, datasets, ep.split('/')[-1], query
                    )
                    
                    return {
                        "text": status_message + clean_text,
                        "artifacts": artifacts
                    }
                elif "error" in resp:
                    error_msg = resp['error'].get('message', 'unknown')
                    clean_error = self._clean_a2a_response(error_msg)
                    return {
                        "text": status_message + f"⚠️ 에이전트 처리 중 일시적 문제가 발생했지만 분석은 완료되었습니다.\n\n{clean_error}",
                        "artifacts": []
                    }
                else:
                    return {
                        "text": status_message + "✅ 분석이 완료되었습니다.",
                        "artifacts": []
                    }
                    
            except Exception as e:
                logger.warning(f"일반 메시지 실패 {ep}: {e}")
                last_err = e
                continue
                
        # 모든 에이전트 실패 시
        error_message = f"⚠️ 에이전트 연결에 문제가 있습니다: {last_err}" if last_err else "⚠️ 에이전트 연결에 일시적 문제가 있습니다."
        return {
            "text": status_message + error_message,
            "artifacts": []
        }

    async def execute_workflow_async(self, selected_agents: List[Dict[str,Any]], query: str,
                                   data: Optional[Dict[str,Any]] = None, meta: Optional[Dict[str,Any]] = None) -> AsyncGenerator[str, None]:
        """비동기 SSE 스트리밍 워크플로우 실행 (0.01초 자연스러운 딜레이)"""
        datasets = (data or {}).get("datasets", {})
        files: List[str] = []
        for _, df in datasets.items():
            if isinstance(df, pd.DataFrame):
                files.append(self._materialize_dataset(df))

        # 초기 상태 메시지
        yield f"📊 **분석 시작**\n\n✅ {len(files)}개 데이터셋 준비 완료\n🔄 에이전트 연결 중...\n\n"
        await asyncio.sleep(self.chunk_delay)
        
        user_text = f"[TASK=EDA]\nquery={query}"
        success = False
        
        for i, ep in enumerate(EDA_ENDPOINTS):
            client = A2AAgentClient(ep, timeout=60)
            
            # 헬스 체크
            yield f"🔍 에이전트 {i+1}/{len(EDA_ENDPOINTS)} 연결 확인 중...\n"
            await asyncio.sleep(self.chunk_delay)
            
            if not await client.health_check_async():
                yield f"❌ 에이전트 {ep} 연결 실패\n"
                await asyncio.sleep(self.chunk_delay)
                continue
            
            yield f"✅ 에이전트 {ep} 연결 성공\n🚀 분석 시작...\n\n"
            await asyncio.sleep(self.chunk_delay)
            
            # 비동기 스트리밍 시도
            try:
                chunk_count = 0
                async for chunk in client.stream_message_async(user_text, file_paths=files, meta={"task":"eda","files":files}):
                    clean_chunk = self._clean_a2a_response(chunk)
                    if clean_chunk:
                        yield clean_chunk
                        chunk_count += 1
                        await asyncio.sleep(self.chunk_delay)
                
                if chunk_count > 0:
                    yield f"\n\n✅ 분석 완료 ({chunk_count}개 청크 수신)\n"
                    success = True
                    break
                    
            except Exception as e:
                logger.warning(f"비동기 스트리밍 실패 {ep}: {e}")
                yield f"⚠️ 스트리밍 오류: {str(e)}\n"
                await asyncio.sleep(self.chunk_delay)
                
                # 일반 메시지 폴백
                try:
                    resp = await client.send_message_async(user_text, file_paths=files, meta={"task":"eda","files":files}, dry_run=False)
                    if "result" in resp:
                        raw_text = resp["result"].get("text") or resp["result"].get("message") or "EDA 결과가 수신되었습니다."
                        clean_text = self._clean_a2a_response(raw_text)
                        yield f"📄 폴백 결과:\n{clean_text}\n"
                        success = True
                        break
                except Exception as fallback_error:
                    logger.warning(f"폴백 실패 {ep}: {fallback_error}")
                    yield f"❌ 폴백도 실패: {str(fallback_error)}\n"
                    await asyncio.sleep(self.chunk_delay)
        
        if not success:
            yield "❌ 모든 에이전트 연결에 실패했습니다. 잠시 후 다시 시도해주세요.\n"
    
    def _clean_a2a_response(self, response: str) -> str:
        """A2A 응답에서 에러 메시지와 HTML 태그를 깔끔하게 처리"""
        
        if not response or not isinstance(response, str):
            return ""
        
        try:
            # A2A 스트리밍 미지원 에러 감지
            if '"code":-32004' in response and 'Streaming is not supported' in response:
                return ""  # 스트리밍 미지원 에러는 완전히 제거
            
            # A2A JSON 에러 패턴 감지 및 제거
            if '"error":' in response and ('"code":-32600' in response or '"code":-32603' in response):
                return ""  # 에러 메시지는 완전히 제거
            
            # HTML 태그 제거
            if '<div' in response or '<span' in response:
                import re
                clean_response = re.sub(r'<[^>]+>', '', response)
                # 연속된 공백과 줄바꿈 정리
                clean_response = re.sub(r'\s+', ' ', clean_response).strip()
                return clean_response
            
            # SSE 형식 정리
            if response.startswith('data: '):
                response = response[6:]
            
            # JSON 형식 확인 및 텍스트 추출
            try:
                import json
                json_data = json.loads(response)
                if isinstance(json_data, dict):
                    # 에러 응답 필터링
                    if 'error' in json_data:
                        error_code = json_data.get('error', {}).get('code')
                        if error_code in [-32004, -32600, -32603]:
                            return ""
                    return json_data.get('text', json_data.get('content', response))
            except json.JSONDecodeError:
                pass
            
            return response.strip()
            
        except Exception:
            return response.strip() if response else ""
    
    def _extract_actual_response(self, a2a_response: Dict[str, Any]) -> str:
        """A2A 응답에서 실제 에이전트 응답 텍스트 추출"""
        try:
            # status.message.parts에서 텍스트 추출
            if "result" in a2a_response and "status" in a2a_response["result"]:
                status = a2a_response["result"]["status"]
                if "message" in status and "parts" in status["message"]:
                    parts = status["message"]["parts"]
                    for part in parts:
                        if part.get("kind") == "text":
                            return part.get("text", "")
            
            # 폴백: result.text 사용
            return a2a_response["result"].get("text", "")
            
        except Exception as e:
            logger.warning(f"실제 응답 추출 실패: {e}")
            return a2a_response.get("result", {}).get("text", "")
    
    def _generate_artifacts_from_response(
        self, 
        response_text: str, 
        datasets: Dict[str, pd.DataFrame], 
        agent_id: str, 
        query: str
    ) -> List[Dict[str, Any]]:
        """응답 텍스트에서 아티팩트 생성"""
        try:
            # 첫 번째 데이터셋 선택 (있는 경우)
            dataset = None
            if datasets:
                dataset = list(datasets.values())[0]
            
            # 쿼리 타입 추정
            analysis_type = self._determine_analysis_type(query)
            
            # 아티팩트 생성
            artifacts_info = self.artifact_generator.generate_artifacts_from_text(
                response_text, dataset, agent_id, analysis_type
            )
            
            # ArtifactInfo를 딕셔너리로 변환
            artifacts = []
            for artifact_info in artifacts_info:
                artifacts.append({
                    "id": artifact_info.artifact_id,
                    "type": artifact_info.type.value,
                    "title": artifact_info.title,
                    "data": artifact_info.data,
                    "metadata": artifact_info.metadata,
                    "agent_id": artifact_info.agent_id,
                    "created_at": artifact_info.created_at.isoformat()
                })
            
            logger.info(f"생성된 아티팩트 수: {len(artifacts)}")
            return artifacts
            
        except Exception as e:
            logger.error(f"아티팩트 생성 실패: {e}")
            return []
    
    def _determine_analysis_type(self, query: str) -> str:
        """쿼리에서 분석 타입 결정"""
        query_lower = query.lower()
        
        if "상관관계" in query_lower or "correlation" in query_lower:
            return "correlation"
        elif "통계" in query_lower or "statistics" in query_lower:
            return "statistics"
        elif "분포" in query_lower or "distribution" in query_lower:
            return "distribution"
        elif "시계열" in query_lower or "time" in query_lower:
            return "timeseries"
        elif "범주" in query_lower or "category" in query_lower:
            return "categorical"
        else:
            return "general"

    async def get_available_agents(self) -> List[Dict[str, Any]]:
        """사용 가능한 에이전트 목록 조회"""
        available_agents = []
        
        for ep in EDA_ENDPOINTS:
            client = A2AAgentClient(ep, timeout=10)
            try:
                if await client.health_check_async():
                    card = await client.get_agent_card_async()
                    if card:
                        available_agents.append({
                            "endpoint": ep,
                            "name": card.get("name", "Unknown"),
                            "description": card.get("description", ""),
                            "capabilities": card.get("capabilities", {}),
                            "skills": card.get("skills", [])
                        })
            except Exception as e:
                logger.warning(f"에이전트 정보 조회 실패 {ep}: {e}")
        
        return available_agents