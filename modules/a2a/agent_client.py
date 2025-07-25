"""
A2A Agent Client - A2A SDK 0.2.9 Integration

Based on proven A2ACommunicationProtocol patterns from Universal Engine.
Handles communication with A2A agents using validated patterns.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
import httpx
import json
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class A2AAgentClient:
    """A2A Agent Client with proven Universal Engine patterns"""
    
    def __init__(self, port: int, base_url: str = "http://localhost"):
        """Initialize A2A Agent Client"""
        self.port = port
        self.base_url = base_url
        self.agent_url = f"{base_url}:{port}"
        self.timeout = 300  # 5 minutes
        self.retry_attempts = 3
        self.retry_delays = [5, 15, 30]  # Progressive retry pattern
        
        # Circuit breaker state
        self.circuit_breaker_failures = 0
        self.circuit_breaker_threshold = 5
        self.circuit_breaker_reset_time = None
        
        logger.info(f"Initialized A2A Agent Client for port {port}")
    
    async def execute_task(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute task using validated Universal Engine patterns:
        1. Enhanced agent request with user_expertise_level and domain_context
        2. JSON-RPC 2.0 protocol with proper error handling
        3. Progressive retry (5s → 15s → 30s) on failures
        4. Circuit breaker pattern for resilience
        """
        # Check circuit breaker
        if self._is_circuit_breaker_open():
            raise Exception(f"Circuit breaker is open for agent {self.port}")
        
        # Enhance request with Universal Engine context
        enhanced_request = self._create_enhanced_request(request_data)
        
        # Execute with retry logic
        last_exception = None
        for attempt in range(self.retry_attempts):
            try:
                response = await self._make_request(enhanced_request)
                
                # Reset circuit breaker on success
                self.circuit_breaker_failures = 0
                self.circuit_breaker_reset_time = None
                
                return response
                
            except Exception as e:
                last_exception = e
                logger.warning(f"Attempt {attempt + 1} failed for agent {self.port}: {str(e)}")
                
                if attempt < self.retry_attempts - 1:
                    delay = self.retry_delays[attempt]
                    logger.info(f"Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
        
        # All attempts failed - trigger circuit breaker
        self._trigger_circuit_breaker()
        raise Exception(f"All retry attempts failed for agent {self.port}: {str(last_exception)}")
    
    def _create_enhanced_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create enhanced request with Universal Engine context patterns"""
        enhanced_request = {
            "jsonrpc": "2.0",
            "method": "process_request",
            "id": self._generate_request_id(),
            "params": {
                **request_data,
                "user_expertise_level": "intermediate",  # Can be customized
                "domain_context": self._infer_domain_context(request_data),
                "collaboration_mode": "enhanced",
                "timestamp": datetime.now().isoformat(),
                "agent_port": self.port,
                "universal_engine_context": True
            }
        }
        
        return enhanced_request
    
    def _infer_domain_context(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Infer domain context from request data"""
        context = {
            "data_science": True,
            "analysis_type": request_data.get("analysis_type", "general"),
            "data_format": "structured"
        }
        
        # Analyze request for specific domain hints
        query = request_data.get("query", "").lower()
        if any(word in query for word in ["time", "series", "temporal", "date"]):
            context["temporal_analysis"] = True
        if any(word in query for word in ["predict", "model", "ml", "machine learning"]):
            context["machine_learning"] = True
        if any(word in query for word in ["visualize", "plot", "chart", "graph"]):
            context["visualization"] = True
        
        return context
    
    async def _make_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make HTTP request to A2A agent"""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(
                    self.agent_url,
                    json=request_data,
                    headers={
                        "Content-Type": "application/json",
                        "User-Agent": "CherryAI-Streamlit-Platform/1.0"
                    }
                )
                
                response.raise_for_status()
                
                # Parse JSON-RPC response
                response_data = response.json()
                
                # Handle JSON-RPC errors
                if "error" in response_data:
                    error_info = response_data["error"]
                    raise Exception(f"Agent error: {error_info.get('message', 'Unknown error')}")
                
                return response_data.get("result", {})
                
            except httpx.TimeoutException:
                raise Exception(f"Request timeout for agent {self.port}")
            except httpx.HTTPStatusError as e:
                raise Exception(f"HTTP error {e.response.status_code} for agent {self.port}")
            except json.JSONDecodeError:
                raise Exception(f"Invalid JSON response from agent {self.port}")
            except Exception as e:
                raise Exception(f"Request failed for agent {self.port}: {str(e)}")
    
    async def validate_agent_endpoint(self) -> Dict[str, Any]:
        """Validate /.well-known/agent.json endpoint (proven pattern)"""
        try:
            well_known_url = f"{self.agent_url}/.well-known/agent.json"
            
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.get(well_known_url)
                response.raise_for_status()
                
                agent_info = response.json()
                
                # Validate required fields
                required_fields = ["name", "version", "capabilities", "endpoints"]
                for field in required_fields:
                    if field not in agent_info:
                        raise Exception(f"Missing required field: {field}")
                
                logger.info(f"Successfully validated agent {self.port}: {agent_info['name']}")
                return agent_info
                
        except Exception as e:
            logger.error(f"Agent validation failed for port {self.port}: {str(e)}")
            raise Exception(f"Agent validation failed: {str(e)}")
    
    async def health_check(self) -> bool:
        """Perform health check on the agent"""
        try:
            health_request = {
                "jsonrpc": "2.0",
                "method": "health_check",
                "id": self._generate_request_id(),
                "params": {}
            }
            
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.post(self.agent_url, json=health_request)
                response.raise_for_status()
                
                response_data = response.json()
                return response_data.get("result", {}).get("status") == "healthy"
                
        except Exception as e:
            logger.warning(f"Health check failed for agent {self.port}: {str(e)}")
            return False
    
    def _is_circuit_breaker_open(self) -> bool:
        """Check if circuit breaker is open"""
        if self.circuit_breaker_failures < self.circuit_breaker_threshold:
            return False
        
        if self.circuit_breaker_reset_time is None:
            return True
        
        # Check if enough time has passed to try again
        return datetime.now() < self.circuit_breaker_reset_time
    
    def _trigger_circuit_breaker(self):
        """Trigger circuit breaker after multiple failures"""
        self.circuit_breaker_failures += 1
        
        if self.circuit_breaker_failures >= self.circuit_breaker_threshold:
            # Set reset time to 5 minutes from now
            self.circuit_breaker_reset_time = datetime.now() + timedelta(minutes=5)
            logger.error(f"Circuit breaker opened for agent {self.port}")
    
    def _generate_request_id(self) -> str:
        """Generate unique request ID"""
        return f"req_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.port}"
    
    def get_agent_capabilities(self) -> Dict[str, List[str]]:
        """Get agent capabilities mapping"""
        # This is based on the proven Universal Engine agent mapping
        capabilities_map = {
            8306: ["data_cleaning", "missing_values", "outliers", "data_validation"],
            8307: ["data_loading", "file_formats", "encoding", "data_import"],
            8308: ["visualization", "charts", "plots", "interactive_graphics"],
            8309: ["data_wrangling", "transformation", "reshaping", "feature_engineering"],
            8310: ["feature_engineering", "feature_selection", "dimensionality_reduction"],
            8311: ["sql_queries", "database_operations", "data_extraction"],
            8312: ["exploratory_analysis", "statistics", "pattern_discovery"],
            8313: ["machine_learning", "model_training", "prediction", "automl"],
            8314: ["model_management", "experiment_tracking", "version_control"],
            8315: ["pandas_operations", "data_manipulation", "analysis"]
        }
        
        return capabilities_map.get(self.port, [])
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get basic agent information"""
        agent_names = {
            8306: "Data Cleaning Agent",
            8307: "Data Loader Agent", 
            8308: "Data Visualization Agent",
            8309: "Data Wrangling Agent",
            8310: "Feature Engineering Agent",
            8311: "SQL Database Agent",
            8312: "EDA Tools Agent",
            8313: "H2O ML Agent",
            8314: "MLflow Tools Agent",
            8315: "Pandas Analyst Agent"
        }
        
        return {
            "port": self.port,
            "name": agent_names.get(self.port, f"Agent {self.port}"),
            "url": self.agent_url,
            "capabilities": self.get_agent_capabilities(),
            "status": "unknown"  # Will be updated by health checks
        }