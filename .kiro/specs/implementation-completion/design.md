# 🔧 LLM-First Universal Engine 구현 완성 설계 문서

## 📋 개요

이 설계 문서는 현재 구현된 LLM-First Universal Engine의 **실제 상황을 기반으로** 완성해야 할 구체적인 설계 방안을 정의합니다.

### 현재 구현 상태 분석
- **✅ 완성된 부분**: 26개 컴포넌트 구조, 기본 초기화, 시스템 통합
- **⚠️ 미완성 부분**: 19개 메서드 인터페이스, 의존성 모듈, 레거시 정리
- **🎯 목표**: 인터페이스 완성 + 레거시 정리 + 품질 보증

### 핵심 설계 원칙
- **점진적 완성**: 기존 구조를 유지하면서 누락된 부분만 추가
- **최소 변경**: 현재 동작하는 부분은 건드리지 않음
- **품질 우선**: 완성과 동시에 품질 보증
- **실용적 접근**: 이론보다는 실제 동작하는 구현

## 🏗️ 완성 아키텍처 설계

### 1. 메서드 인터페이스 완성 설계

#### 1.1 UniversalQueryProcessor 완성
```python
class UniversalQueryProcessor:
    """
    현재 상태: 기본 구조 완성, process_query 구현됨
    완성 필요: initialize, get_status 메서드
    """
    
    def __init__(self):
        # 기존 초기화 코드 유지
        self.meta_reasoning_engine = MetaReasoningEngine()
        self.dynamic_context_discovery = DynamicContextDiscovery()
        self.adaptive_user_understanding = AdaptiveUserUnderstanding()
        self.universal_intent_detection = UniversalIntentDetection()
        self.is_initialized = False
        self.status = {"state": "created", "components": {}}
    
    async def initialize(self) -> Dict[str, Any]:
        """
        시스템 초기화 및 의존성 검증
        - 모든 하위 컴포넌트 초기화
        - 의존성 검증
        - 상태 업데이트
        """
        try:
            # 1. 하위 컴포넌트 초기화
            await self.meta_reasoning_engine.initialize()
            await self.dynamic_context_discovery.initialize()
            await self.adaptive_user_understanding.initialize()
            await self.universal_intent_detection.initialize()
            
            # 2. 시스템 상태 업데이트
            self.is_initialized = True
            self.status["state"] = "initialized"
            self.status["timestamp"] = datetime.now().isoformat()
            
            return {
                "status": "success",
                "message": "UniversalQueryProcessor initialized successfully",
                "components_initialized": 4,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.status["state"] = "initialization_failed"
            self.status["error"] = str(e)
            return {
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def get_status(self) -> Dict[str, Any]:
        """
        현재 시스템 상태 반환
        - 초기화 상태
        - 컴포넌트 상태
        - 성능 메트릭
        """
        component_status = {}
        
        # 각 컴포넌트 상태 수집
        if hasattr(self.meta_reasoning_engine, 'get_status'):
            component_status["meta_reasoning"] = await self.meta_reasoning_engine.get_status()
        
        if hasattr(self.dynamic_context_discovery, 'get_status'):
            component_status["context_discovery"] = await self.dynamic_context_discovery.get_status()
        
        return {
            "processor_status": self.status,
            "is_initialized": self.is_initialized,
            "component_status": component_status,
            "system_health": "healthy" if self.is_initialized else "not_ready",
            "timestamp": datetime.now().isoformat()
        }
```

#### 1.2 MetaReasoningEngine 완성
```python
class MetaReasoningEngine:
    """
    현재 상태: analyze_request 구현됨
    완성 필요: perform_meta_reasoning, assess_analysis_quality
    """
    
    async def perform_meta_reasoning(self, query: str, context: Dict) -> Dict[str, Any]:
        """
        완전한 메타 추론 프로세스 실행
        DeepSeek-R1 기반 4단계 추론 + 자가 평가
        """
        reasoning_result = {
            "reasoning_id": f"meta_reasoning_{int(datetime.now().timestamp())}",
            "query": query,
            "context": context,
            "stages": {},
            "final_assessment": {},
            "confidence_score": 0.0
        }
        
        try:
            # 1단계: 초기 관찰
            stage1 = await self._perform_initial_observation(query, context)
            reasoning_result["stages"]["initial_observation"] = stage1
            
            # 2단계: 다각도 분석  
            stage2 = await self._perform_multi_perspective_analysis(stage1, query, context)
            reasoning_result["stages"]["multi_perspective"] = stage2
            
            # 3단계: 자가 검증
            stage3 = await self._perform_self_verification(stage2)
            reasoning_result["stages"]["self_verification"] = stage3
            
            # 4단계: 적응적 응답 전략
            stage4 = await self._determine_adaptive_strategy(stage3, context)
            reasoning_result["stages"]["adaptive_strategy"] = stage4
            
            # 최종 평가
            final_assessment = await self.assess_analysis_quality(reasoning_result)
            reasoning_result["final_assessment"] = final_assessment
            reasoning_result["confidence_score"] = final_assessment.get("confidence", 0.0)
            
            return reasoning_result
            
        except Exception as e:
            reasoning_result["error"] = str(e)
            reasoning_result["status"] = "failed"
            return reasoning_result
    
    async def assess_analysis_quality(self, analysis_result: Dict) -> Dict[str, Any]:
        """
        분석 품질 평가 및 개선 제안
        메타 보상 패턴 적용
        """
        quality_assessment = {
            "assessment_id": f"quality_assessment_{int(datetime.now().timestamp())}",
            "metrics": {},
            "overall_score": 0.0,
            "strengths": [],
            "weaknesses": [],
            "improvement_suggestions": []
        }
        
        try:
            # 5가지 평가 기준 적용
            metrics = {}
            
            # 1. 정확성 평가
            accuracy_score = await self._assess_accuracy(analysis_result)
            metrics["accuracy"] = accuracy_score
            
            # 2. 완전성 평가
            completeness_score = await self._assess_completeness(analysis_result)
            metrics["completeness"] = completeness_score
            
            # 3. 적절성 평가
            appropriateness_score = await self._assess_appropriateness(analysis_result)
            metrics["appropriateness"] = appropriateness_score
            
            # 4. 명확성 평가
            clarity_score = await self._assess_clarity(analysis_result)
            metrics["clarity"] = clarity_score
            
            # 5. 실용성 평가
            practicality_score = await self._assess_practicality(analysis_result)
            metrics["practicality"] = practicality_score
            
            # 전체 점수 계산
            overall_score = sum(metrics.values()) / len(metrics)
            
            quality_assessment["metrics"] = metrics
            quality_assessment["overall_score"] = overall_score
            quality_assessment["confidence"] = min(overall_score, 0.95)  # 최대 95%
            
            # 강점과 약점 식별
            for metric, score in metrics.items():
                if score >= 0.8:
                    quality_assessment["strengths"].append(f"High {metric}: {score:.2f}")
                elif score < 0.6:
                    quality_assessment["weaknesses"].append(f"Low {metric}: {score:.2f}")
            
            # 개선 제안
            if overall_score < 0.8:
                quality_assessment["improvement_suggestions"] = await self._generate_improvement_suggestions(metrics)
            
            return quality_assessment
            
        except Exception as e:
            quality_assessment["error"] = str(e)
            quality_assessment["overall_score"] = 0.0
            return quality_assessment
```

#### 1.3 DynamicContextDiscovery 완성
```python
class DynamicContextDiscovery:
    """
    현재 상태: discover_context 구현됨
    완성 필요: analyze_data_characteristics, detect_domain
    """
    
    async def analyze_data_characteristics(self, data: Any) -> Dict[str, Any]:
        """
        데이터 특성 자동 분석
        - 데이터 타입 및 구조 분석
        - 통계적 특성 추출
        - 패턴 및 이상치 감지
        """
        characteristics = {
            "analysis_id": f"data_analysis_{int(datetime.now().timestamp())}",
            "data_type": None,
            "structure": {},
            "statistics": {},
            "patterns": [],
            "anomalies": [],
            "quality_score": 0.0
        }
        
        try:
            # 1. 데이터 타입 식별
            if hasattr(data, 'shape'):  # DataFrame-like
                characteristics["data_type"] = "tabular"
                characteristics["structure"] = {
                    "rows": data.shape[0] if len(data.shape) > 0 else 0,
                    "columns": data.shape[1] if len(data.shape) > 1 else 0,
                    "column_names": list(data.columns) if hasattr(data, 'columns') else []
                }
                
                # 통계적 특성
                if hasattr(data, 'describe'):
                    stats = data.describe()
                    characteristics["statistics"] = stats.to_dict() if hasattr(stats, 'to_dict') else {}
                
            elif isinstance(data, (list, tuple)):
                characteristics["data_type"] = "sequence"
                characteristics["structure"] = {
                    "length": len(data),
                    "element_types": list(set(type(item).__name__ for item in data[:10]))
                }
                
            elif isinstance(data, dict):
                characteristics["data_type"] = "dictionary"
                characteristics["structure"] = {
                    "keys": list(data.keys())[:10],
                    "key_count": len(data)
                }
                
            else:
                characteristics["data_type"] = type(data).__name__
            
            # 2. 패턴 감지 (LLM 기반)
            patterns = await self._detect_data_patterns(data, characteristics)
            characteristics["patterns"] = patterns
            
            # 3. 품질 평가
            quality_score = await self._assess_data_quality(data, characteristics)
            characteristics["quality_score"] = quality_score
            
            return characteristics
            
        except Exception as e:
            characteristics["error"] = str(e)
            return characteristics
    
    async def detect_domain(self, data: Any, query: str) -> Dict[str, Any]:
        """
        도메인 컨텍스트 자동 감지
        - 데이터 내용 기반 도메인 추론
        - 쿼리 컨텍스트 분석
        - 도메인별 특성 식별
        """
        domain_detection = {
            "detection_id": f"domain_detection_{int(datetime.now().timestamp())}",
            "detected_domains": [],
            "confidence_scores": {},
            "domain_characteristics": {},
            "recommended_approach": {},
            "uncertainty_areas": []
        }
        
        try:
            # 1. 데이터 기반 도메인 추론
            data_characteristics = await self.analyze_data_characteristics(data)
            
            # 2. LLM 기반 도메인 감지
            domain_analysis_prompt = f"""
            다음 데이터와 쿼리를 분석하여 도메인을 감지해주세요:
            
            데이터 특성: {json.dumps(data_characteristics, indent=2)}
            사용자 쿼리: {query}
            
            다음 관점에서 분석해주세요:
            1. 데이터에서 발견되는 도메인 특성
            2. 쿼리에서 나타나는 도메인 컨텍스트
            3. 가능한 도메인들과 각각의 확신도
            4. 도메인별 분석 접근법 추천
            
            JSON 형식으로 응답해주세요.
            """
            
            # LLM 호출 (실제 구현에서는 self.llm_client 사용)
            llm_response = await self._call_llm_for_domain_detection(domain_analysis_prompt)
            
            # 3. 결과 통합
            if llm_response:
                domain_detection.update(llm_response)
            
            # 4. 불확실성 평가
            uncertainty_areas = await self._identify_uncertainty_areas(domain_detection, data_characteristics)
            domain_detection["uncertainty_areas"] = uncertainty_areas
            
            return domain_detection
            
        except Exception as e:
            domain_detection["error"] = str(e)
            return domain_detection
```

### 2. A2A 통합 컴포넌트 완성 설계

#### 2.1 A2AAgentDiscoverySystem 완성
```python
class A2AAgentDiscoverySystem:
    """
    현재 상태: 기본 구조 완성
    완성 필요: discover_available_agents, validate_agent_endpoint, monitor_agent_health
    """
    
    async def discover_available_agents(self) -> Dict[str, Any]:
        """
        사용 가능한 A2A 에이전트 자동 발견
        포트 8306-8315 스캔 및 에이전트 정보 수집
        """
        discovery_result = {
            "discovery_id": f"agent_discovery_{int(datetime.now().timestamp())}",
            "discovered_agents": {},
            "total_agents": 0,
            "available_agents": 0,
            "failed_agents": [],
            "discovery_time": 0.0
        }
        
        start_time = time.time()
        
        try:
            # 포트 8306-8315 스캔
            for agent_id, port in self.AGENT_PORTS.items():
                endpoint = f"http://localhost:{port}"
                
                try:
                    # 에이전트 엔드포인트 검증
                    validation_result = await self.validate_agent_endpoint(endpoint)
                    
                    if validation_result["is_valid"]:
                        # 에이전트 정보 수집
                        agent_info = await self._collect_agent_info(endpoint)
                        
                        discovery_result["discovered_agents"][agent_id] = {
                            "id": agent_id,
                            "port": port,
                            "endpoint": endpoint,
                            "status": "available",
                            "info": agent_info,
                            "validation": validation_result
                        }
                        discovery_result["available_agents"] += 1
                    else:
                        discovery_result["failed_agents"].append({
                            "agent_id": agent_id,
                            "port": port,
                            "error": validation_result.get("error", "Validation failed")
                        })
                        
                except Exception as e:
                    discovery_result["failed_agents"].append({
                        "agent_id": agent_id,
                        "port": port,
                        "error": str(e)
                    })
                
                discovery_result["total_agents"] += 1
            
            discovery_result["discovery_time"] = time.time() - start_time
            
            # 발견된 에이전트 정보 업데이트
            self.discovered_agents = discovery_result["discovered_agents"]
            
            return discovery_result
            
        except Exception as e:
            discovery_result["error"] = str(e)
            discovery_result["discovery_time"] = time.time() - start_time
            return discovery_result
    
    async def validate_agent_endpoint(self, endpoint: str) -> Dict[str, Any]:
        """
        에이전트 엔드포인트 유효성 검증
        - 연결 가능성 확인
        - A2A 프로토콜 준수 확인
        - 에이전트 카드 정보 검증
        """
        validation_result = {
            "endpoint": endpoint,
            "is_valid": False,
            "checks": {},
            "agent_info": {},
            "validation_time": 0.0
        }
        
        start_time = time.time()
        
        try:
            # 1. 기본 연결 확인
            connection_check = await self._check_connection(endpoint)
            validation_result["checks"]["connection"] = connection_check
            
            if not connection_check["success"]:
                validation_result["validation_time"] = time.time() - start_time
                return validation_result
            
            # 2. 에이전트 카드 확인
            agent_card_check = await self._check_agent_card(endpoint)
            validation_result["checks"]["agent_card"] = agent_card_check
            
            if agent_card_check["success"]:
                validation_result["agent_info"] = agent_card_check["data"]
            
            # 3. A2A 프로토콜 확인
            protocol_check = await self._check_a2a_protocol(endpoint)
            validation_result["checks"]["protocol"] = protocol_check
            
            # 전체 검증 결과
            all_checks_passed = all(
                check["success"] for check in validation_result["checks"].values()
            )
            validation_result["is_valid"] = all_checks_passed
            validation_result["validation_time"] = time.time() - start_time
            
            return validation_result
            
        except Exception as e:
            validation_result["error"] = str(e)
            validation_result["validation_time"] = time.time() - start_time
            return validation_result
    
    async def monitor_agent_health(self, agent_id: str) -> Dict[str, Any]:
        """
        에이전트 상태 모니터링
        - 응답 시간 측정
        - 리소스 사용량 확인
        - 오류율 추적
        """
        if agent_id not in self.discovered_agents:
            return {
                "agent_id": agent_id,
                "status": "not_found",
                "error": "Agent not in discovered agents list"
            }
        
        agent = self.discovered_agents[agent_id]
        health_result = {
            "agent_id": agent_id,
            "endpoint": agent["endpoint"],
            "health_status": "unknown",
            "metrics": {},
            "alerts": [],
            "monitoring_time": 0.0
        }
        
        start_time = time.time()
        
        try:
            # 1. 응답 시간 측정
            response_time = await self._measure_response_time(agent["endpoint"])
            health_result["metrics"]["response_time"] = response_time
            
            # 2. 상태 확인
            status_check = await self._check_agent_status(agent["endpoint"])
            health_result["metrics"]["status"] = status_check
            
            # 3. 건강 상태 평가
            if response_time < 1.0 and status_check.get("healthy", False):
                health_result["health_status"] = "healthy"
            elif response_time < 3.0:
                health_result["health_status"] = "degraded"
                health_result["alerts"].append("High response time")
            else:
                health_result["health_status"] = "unhealthy"
                health_result["alerts"].append("Very high response time")
            
            health_result["monitoring_time"] = time.time() - start_time
            
            return health_result
            
        except Exception as e:
            health_result["error"] = str(e)
            health_result["health_status"] = "error"
            health_result["monitoring_time"] = time.time() - start_time
            return health_result
```

### 3. 누락된 의존성 구현 설계

#### 3.1 LLMFactory 모듈 구현
```python
# core/universal_engine/llm_factory.py
class LLMFactory:
    """
    LLM 클라이언트 팩토리
    다양한 LLM 제공자 지원 및 통합 인터페이스 제공
    """
    
    SUPPORTED_PROVIDERS = {
        "ollama": "langchain_ollama",
        "openai": "langchain_openai", 
        "anthropic": "langchain_anthropic",
        "local": "local_llm"
    }
    
    @staticmethod
    def create_llm_client(config: Dict = None) -> Any:
        """
        설정에 따른 LLM 클라이언트 생성
        """
        if config is None:
            config = LLMFactory._get_default_config()
        
        provider = config.get("provider", "ollama")
        
        if provider == "ollama":
            return LLMFactory._create_ollama_client(config)
        elif provider == "openai":
            return LLMFactory._create_openai_client(config)
        elif provider == "anthropic":
            return LLMFactory._create_anthropic_client(config)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
    
    @staticmethod
    def get_available_models() -> List[str]:
        """
        사용 가능한 모델 목록 반환
        """
        models = []
        
        # Ollama 모델 확인
        try:
            from langchain_ollama import OllamaLLM
            # 실제 구현에서는 Ollama API 호출
            models.extend(["llama3.1", "qwen2.5", "deepseek-r1"])
        except ImportError:
            pass
        
        return models
    
    @staticmethod
    def validate_model_config(config: Dict) -> bool:
        """
        모델 설정 유효성 검증
        """
        required_fields = ["provider"]
        
        for field in required_fields:
            if field not in config:
                return False
        
        provider = config["provider"]
        if provider not in LLMFactory.SUPPORTED_PROVIDERS:
            return False
        
        return True
```

### 4. 레거시 하드코딩 제거 설계

#### 4.1 하드코딩 패턴 제거 전략
```python
# 제거 대상 파일별 전략

# 실제 사용되는 파일들만 대상으로 선별
LEGACY_PATTERNS_TO_REMOVE = {
    "core/query_processing/domain_extractor.py": [
        {
            "pattern": '_initialize_domain_patterns()',
            "replacement": "await self.dynamic_context_discovery.detect_domain(data, query)",
            "description": "하드코딩된 도메인 패턴을 LLM 기반 동적 감지로 대체"
        },
        {
            "pattern": '_initialize_methodology_database()',
            "replacement": "await self.meta_reasoning_engine.perform_meta_reasoning(query, context)",
            "description": "하드코딩된 방법론 DB를 LLM 기반 메타 추론으로 대체"
        }
    ],
    
    "core/orchestrator/planning_engine.py": [
        {
            "pattern": "if domain == 'semiconductor':",
            "replacement": "domain_analysis = await self.universal_engine.detect_domain(data, query)",
            "description": "하드코딩된 도메인 분기를 동적 도메인 감지로 대체"
        },
        {
            "pattern": "data_loader = next((agent for agent in available_agents if agent.id == 'data_loader'), None)",
            "replacement": "selected_agents = await self.a2a_discovery.discover_available_agents()",
            "description": "하드코딩된 에이전트 선택을 동적 에이전트 발견으로 대체"
        }
    ]
}

# Legacy 파일 처리
LEGACY_FILES_TO_MOVE = [
    "cherry_ai_legacy.py"  # 실제 사용되지 않으므로 legacy/ 폴더로 이동
]
```

## 🎯 구현 우선순위

### Phase 1: 핵심 인터페이스 완성 (3-5일)
1. **UniversalQueryProcessor**: `initialize`, `get_status` 구현
2. **MetaReasoningEngine**: `perform_meta_reasoning`, `assess_analysis_quality` 구현
3. **DynamicContextDiscovery**: `analyze_data_characteristics`, `detect_domain` 구현

### Phase 2: A2A 통합 완성 (2-3일)
1. **A2AAgentDiscoverySystem**: 3개 누락 메서드 구현
2. **A2AWorkflowOrchestrator**: 3개 누락 메서드 구현
3. **의존성 해결**: `llm_factory` 모듈 구현

### Phase 3: 레거시 정리 (2-3일)
1. **하드코딩 패턴 제거**: 31개 위반 사항 정리
2. **LLM 기반 로직 대체**: 동적 처리 로직 구현
3. **코드 품질 개선**: 리팩토링 및 정리

### Phase 4: 품질 보증 (2-3일)
1. **종합 테스트**: 모든 컴포넌트 검증
2. **성능 최적화**: 응답 시간 및 리소스 사용량 최적화
3. **문서화**: 구현 완성 문서 업데이트

이 설계를 따라 구현하면 현재 99% 완성된 Universal Engine을 100% 완성할 수 있습니다.