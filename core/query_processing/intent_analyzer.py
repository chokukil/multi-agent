"""
Multi-Perspective Intent Analyzer for CherryAI

This module provides advanced intent analysis from multiple expert perspectives,
enabling more nuanced understanding of user queries.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from core.llm_factory import create_llm_instance
from core.utils.logging import get_logger
from .intelligent_query_processor import QueryType, IntentAnalysis

logger = get_logger(__name__)

class AnalysisPerspective(Enum):
    """Different analytical perspectives"""
    DATA_SCIENTIST = "data_scientist"
    DOMAIN_EXPERT = "domain_expert"
    TECHNICAL_IMPLEMENTER = "technical_implementer"
    BUSINESS_ANALYST = "business_analyst"
    END_USER = "end_user"

class QueryComplexity(Enum):
    """Query complexity levels"""
    SIMPLE = "simple"           # 0.0 - 0.3
    MODERATE = "moderate"       # 0.3 - 0.6
    COMPLEX = "complex"         # 0.6 - 0.8
    VERY_COMPLEX = "very_complex"  # 0.8 - 1.0

class UrgencyLevel(Enum):
    """Urgency classification"""
    LOW = "low"                 # 0.0 - 0.3
    MEDIUM = "medium"           # 0.3 - 0.7
    HIGH = "high"               # 0.7 - 0.9
    CRITICAL = "critical"       # 0.9 - 1.0

@dataclass
class PerspectiveAnalysis:
    """Analysis from a specific perspective"""
    perspective: AnalysisPerspective
    primary_concerns: List[str]
    methodology_suggestions: List[str]
    potential_challenges: List[str]
    success_criteria: List[str]
    estimated_effort: float  # 0.0 - 1.0
    confidence_level: float  # 0.0 - 1.0

@dataclass
class DetailedIntentAnalysis:
    """Enhanced intent analysis with multiple perspectives"""
    primary_intent: str
    secondary_intents: List[str]
    query_type: QueryType
    complexity_level: QueryComplexity
    urgency_level: UrgencyLevel
    perspectives: Dict[AnalysisPerspective, PerspectiveAnalysis]
    overall_confidence: float
    execution_priority: int  # 1-10
    estimated_timeline: str
    critical_dependencies: List[str]
    
class MultiPerspectiveIntentAnalyzer:
    """
    Advanced intent analyzer that examines queries from multiple expert perspectives
    """
    
    def __init__(self):
        self.llm = create_llm_instance()
        logger.info("🧠 MultiPerspectiveIntentAnalyzer initialized")
    
    async def analyze_intent_comprehensive(self, query: str, data_context: Optional[Dict] = None) -> DetailedIntentAnalysis:
        """
        Perform comprehensive intent analysis from multiple perspectives
        
        Args:
            query: User query to analyze
            data_context: Available data context
            
        Returns:
            DetailedIntentAnalysis with multi-perspective insights
        """
        logger.info(f"🔍 Starting comprehensive intent analysis for: {query[:100]}...")
        
        try:
            # Step 1: Basic intent classification
            basic_analysis = await self._classify_basic_intent(query, data_context)
            
            # Step 2: Multi-perspective analysis
            perspectives = await self._analyze_multiple_perspectives(query, basic_analysis, data_context)
            
            # Step 3: Synthesis and priority setting
            synthesis = await self._synthesize_perspectives(query, basic_analysis, perspectives)
            
            return synthesis
            
        except Exception as e:
            logger.error(f"❌ Comprehensive intent analysis failed: {str(e)}")
            return self._create_fallback_analysis(query)
    
    async def _classify_basic_intent(self, query: str, data_context: Optional[Dict] = None) -> Dict:
        """Classify the basic intent and characteristics of the query"""
        
        prompt = f"""
        You are an expert query analyst. Analyze the following user query and classify its basic characteristics.
        
        Query: "{query}"
        Data Context: {json.dumps(data_context, indent=2) if data_context else "None"}
        
        Analyze the query and identify:
        1. Primary intent (what the user fundamentally wants to achieve)
        2. Secondary intents (additional goals or sub-tasks)
        3. Query type (analytical, diagnostic, predictive, etc.)
        4. Complexity level (simple, moderate, complex, very_complex)
        5. Urgency level (low, medium, high, critical)
        6. Critical dependencies (what is needed to fulfill this request)
        
        IMPORTANT: Respond ONLY with valid JSON in this exact format:
        {{
            "primary_intent": "Clear description of main goal",
            "secondary_intents": ["secondary_goal_1", "secondary_goal_2"],
            "query_type": "analytical",
            "complexity_level": "complex",
            "urgency_level": "high",
            "critical_dependencies": ["dependency_1", "dependency_2"],
            "estimated_timeline": "2-3 days",
            "execution_priority": 7
        }}
        """
        
        try:
            response = await asyncio.to_thread(self.llm.invoke, prompt)
            response_text = self._clean_json_response(response)
            return json.loads(response_text)
            
        except Exception as e:
            logger.error(f"Basic intent classification failed: {str(e)}")
            return {
                "primary_intent": "Data analysis request",
                "secondary_intents": [],
                "query_type": "analytical",
                "complexity_level": "moderate",
                "urgency_level": "medium",
                "critical_dependencies": ["data access"],
                "estimated_timeline": "1-2 days",
                "execution_priority": 5
            }
    
    async def _analyze_multiple_perspectives(self, query: str, basic_analysis: Dict, data_context: Optional[Dict] = None) -> Dict[AnalysisPerspective, PerspectiveAnalysis]:
        """Analyze the query from multiple expert perspectives"""
        
        perspectives = {}
        
        # Analyze from each perspective
        perspective_tasks = [
            self._analyze_data_scientist_perspective(query, basic_analysis, data_context),
            self._analyze_domain_expert_perspective(query, basic_analysis, data_context),
            self._analyze_technical_implementer_perspective(query, basic_analysis, data_context),
            self._analyze_business_analyst_perspective(query, basic_analysis, data_context),
            self._analyze_end_user_perspective(query, basic_analysis, data_context)
        ]
        
        # Execute all perspective analyses in parallel
        perspective_results = await asyncio.gather(*perspective_tasks, return_exceptions=True)
        
        # Map results to perspectives
        perspective_types = [
            AnalysisPerspective.DATA_SCIENTIST,
            AnalysisPerspective.DOMAIN_EXPERT,
            AnalysisPerspective.TECHNICAL_IMPLEMENTER,
            AnalysisPerspective.BUSINESS_ANALYST,
            AnalysisPerspective.END_USER
        ]
        
        for i, result in enumerate(perspective_results):
            if not isinstance(result, Exception):
                perspectives[perspective_types[i]] = result
            else:
                logger.error(f"Perspective analysis failed for {perspective_types[i]}: {result}")
                perspectives[perspective_types[i]] = self._create_fallback_perspective_analysis(perspective_types[i])
        
        return perspectives
    
    async def _analyze_data_scientist_perspective(self, query: str, basic_analysis: Dict, data_context: Optional[Dict] = None) -> PerspectiveAnalysis:
        """Analyze from data scientist perspective"""
        
        prompt = f"""
        You are a senior data scientist. Analyze this query from your professional perspective.
        
        Query: "{query}"
        Primary Intent: {basic_analysis.get('primary_intent', '')}
        Query Type: {basic_analysis.get('query_type', '')}
        
        As a data scientist, consider:
        1. What are your primary concerns about this request?
        2. What methodologies would you suggest?
        3. What challenges do you anticipate?
        4. How would you measure success?
        5. What's your estimated effort level (0.0-1.0)?
        6. How confident are you in addressing this (0.0-1.0)?
        
        IMPORTANT: Respond ONLY with valid JSON:
        {{
            "primary_concerns": ["concern_1", "concern_2"],
            "methodology_suggestions": ["method_1", "method_2"],
            "potential_challenges": ["challenge_1", "challenge_2"],
            "success_criteria": ["criteria_1", "criteria_2"],
            "estimated_effort": 0.7,
            "confidence_level": 0.8
        }}
        """
        
        try:
            response = await asyncio.to_thread(self.llm.invoke, prompt)
            response_text = self._clean_json_response(response)
            data = json.loads(response_text)
            
            return PerspectiveAnalysis(
                perspective=AnalysisPerspective.DATA_SCIENTIST,
                primary_concerns=data.get("primary_concerns", []),
                methodology_suggestions=data.get("methodology_suggestions", []),
                potential_challenges=data.get("potential_challenges", []),
                success_criteria=data.get("success_criteria", []),
                estimated_effort=float(data.get("estimated_effort", 0.5)),
                confidence_level=float(data.get("confidence_level", 0.5))
            )
            
        except Exception as e:
            logger.error(f"Data scientist perspective analysis failed: {str(e)}")
            return self._create_fallback_perspective_analysis(AnalysisPerspective.DATA_SCIENTIST)
    
    async def _analyze_domain_expert_perspective(self, query: str, basic_analysis: Dict, data_context: Optional[Dict] = None) -> PerspectiveAnalysis:
        """Analyze from domain expert perspective"""
        
        prompt = f"""
        You are a domain expert with deep industry knowledge. Analyze this query from your specialized perspective.
        
        Query: "{query}"
        Primary Intent: {basic_analysis.get('primary_intent', '')}
        
        As a domain expert, consider:
        1. What domain-specific concerns do you have?
        2. What industry-standard methodologies apply?
        3. What domain-specific challenges exist?
        4. How would you define success in this domain?
        5. What's your estimated effort level (0.0-1.0)?
        6. How confident are you about domain requirements (0.0-1.0)?
        
        IMPORTANT: Respond ONLY with valid JSON:
        {{
            "primary_concerns": ["domain_concern_1", "domain_concern_2"],
            "methodology_suggestions": ["domain_method_1", "domain_method_2"],
            "potential_challenges": ["domain_challenge_1", "domain_challenge_2"],
            "success_criteria": ["domain_criteria_1", "domain_criteria_2"],
            "estimated_effort": 0.6,
            "confidence_level": 0.9
        }}
        """
        
        try:
            response = await asyncio.to_thread(self.llm.invoke, prompt)
            response_text = self._clean_json_response(response)
            data = json.loads(response_text)
            
            return PerspectiveAnalysis(
                perspective=AnalysisPerspective.DOMAIN_EXPERT,
                primary_concerns=data.get("primary_concerns", []),
                methodology_suggestions=data.get("methodology_suggestions", []),
                potential_challenges=data.get("potential_challenges", []),
                success_criteria=data.get("success_criteria", []),
                estimated_effort=float(data.get("estimated_effort", 0.5)),
                confidence_level=float(data.get("confidence_level", 0.5))
            )
            
        except Exception as e:
            logger.error(f"Domain expert perspective analysis failed: {str(e)}")
            return self._create_fallback_perspective_analysis(AnalysisPerspective.DOMAIN_EXPERT)
    
    async def _analyze_technical_implementer_perspective(self, query: str, basic_analysis: Dict, data_context: Optional[Dict] = None) -> PerspectiveAnalysis:
        """Analyze from technical implementer perspective"""
        
        prompt = f"""
        You are a technical implementer responsible for making this happen. Analyze from your implementation perspective.
        
        Query: "{query}"
        Primary Intent: {basic_analysis.get('primary_intent', '')}
        
        As a technical implementer, consider:
        1. What are your technical concerns?
        2. What implementation approaches would you suggest?
        3. What technical challenges do you anticipate?
        4. How would you measure implementation success?
        5. What's your estimated implementation effort (0.0-1.0)?
        6. How confident are you about feasibility (0.0-1.0)?
        
        IMPORTANT: Respond ONLY with valid JSON:
        {{
            "primary_concerns": ["tech_concern_1", "tech_concern_2"],
            "methodology_suggestions": ["tech_approach_1", "tech_approach_2"],
            "potential_challenges": ["tech_challenge_1", "tech_challenge_2"],
            "success_criteria": ["tech_criteria_1", "tech_criteria_2"],
            "estimated_effort": 0.8,
            "confidence_level": 0.7
        }}
        """
        
        try:
            response = await asyncio.to_thread(self.llm.invoke, prompt)
            response_text = self._clean_json_response(response)
            data = json.loads(response_text)
            
            return PerspectiveAnalysis(
                perspective=AnalysisPerspective.TECHNICAL_IMPLEMENTER,
                primary_concerns=data.get("primary_concerns", []),
                methodology_suggestions=data.get("methodology_suggestions", []),
                potential_challenges=data.get("potential_challenges", []),
                success_criteria=data.get("success_criteria", []),
                estimated_effort=float(data.get("estimated_effort", 0.5)),
                confidence_level=float(data.get("confidence_level", 0.5))
            )
            
        except Exception as e:
            logger.error(f"Technical implementer perspective analysis failed: {str(e)}")
            return self._create_fallback_perspective_analysis(AnalysisPerspective.TECHNICAL_IMPLEMENTER)
    
    async def _analyze_business_analyst_perspective(self, query: str, basic_analysis: Dict, data_context: Optional[Dict] = None) -> PerspectiveAnalysis:
        """Analyze from business analyst perspective"""
        
        prompt = f"""
        You are a business analyst focused on value delivery and business outcomes. Analyze from your perspective.
        
        Query: "{query}"
        Primary Intent: {basic_analysis.get('primary_intent', '')}
        
        As a business analyst, consider:
        1. What are your business concerns?
        2. What business methodologies would you suggest?
        3. What business challenges do you see?
        4. How would you measure business success?
        5. What's your estimated business effort (0.0-1.0)?
        6. How confident are you about business value (0.0-1.0)?
        
        IMPORTANT: Respond ONLY with valid JSON:
        {{
            "primary_concerns": ["business_concern_1", "business_concern_2"],
            "methodology_suggestions": ["business_method_1", "business_method_2"],
            "potential_challenges": ["business_challenge_1", "business_challenge_2"],
            "success_criteria": ["business_criteria_1", "business_criteria_2"],
            "estimated_effort": 0.5,
            "confidence_level": 0.8
        }}
        """
        
        try:
            response = await asyncio.to_thread(self.llm.invoke, prompt)
            response_text = self._clean_json_response(response)
            data = json.loads(response_text)
            
            return PerspectiveAnalysis(
                perspective=AnalysisPerspective.BUSINESS_ANALYST,
                primary_concerns=data.get("primary_concerns", []),
                methodology_suggestions=data.get("methodology_suggestions", []),
                potential_challenges=data.get("potential_challenges", []),
                success_criteria=data.get("success_criteria", []),
                estimated_effort=float(data.get("estimated_effort", 0.5)),
                confidence_level=float(data.get("confidence_level", 0.5))
            )
            
        except Exception as e:
            logger.error(f"Business analyst perspective analysis failed: {str(e)}")
            return self._create_fallback_perspective_analysis(AnalysisPerspective.BUSINESS_ANALYST)
    
    async def _analyze_end_user_perspective(self, query: str, basic_analysis: Dict, data_context: Optional[Dict] = None) -> PerspectiveAnalysis:
        """Analyze from end user perspective"""
        
        prompt = f"""
        You are representing the end user who will receive and use the results. Analyze from the user perspective.
        
        Query: "{query}"
        Primary Intent: {basic_analysis.get('primary_intent', '')}
        
        As an end user, consider:
        1. What are your usability concerns?
        2. What approaches would work best for you?
        3. What challenges might you face using the results?
        4. How would you know if this is successful?
        5. What's your estimated learning effort (0.0-1.0)?
        6. How confident are you this will meet your needs (0.0-1.0)?
        
        IMPORTANT: Respond ONLY with valid JSON:
        {{
            "primary_concerns": ["user_concern_1", "user_concern_2"],
            "methodology_suggestions": ["user_approach_1", "user_approach_2"],
            "potential_challenges": ["user_challenge_1", "user_challenge_2"],
            "success_criteria": ["user_criteria_1", "user_criteria_2"],
            "estimated_effort": 0.3,
            "confidence_level": 0.6
        }}
        """
        
        try:
            response = await asyncio.to_thread(self.llm.invoke, prompt)
            response_text = self._clean_json_response(response)
            data = json.loads(response_text)
            
            return PerspectiveAnalysis(
                perspective=AnalysisPerspective.END_USER,
                primary_concerns=data.get("primary_concerns", []),
                methodology_suggestions=data.get("methodology_suggestions", []),
                potential_challenges=data.get("potential_challenges", []),
                success_criteria=data.get("success_criteria", []),
                estimated_effort=float(data.get("estimated_effort", 0.5)),
                confidence_level=float(data.get("confidence_level", 0.5))
            )
            
        except Exception as e:
            logger.error(f"End user perspective analysis failed: {str(e)}")
            return self._create_fallback_perspective_analysis(AnalysisPerspective.END_USER)
    
    async def _synthesize_perspectives(self, query: str, basic_analysis: Dict, perspectives: Dict[AnalysisPerspective, PerspectiveAnalysis]) -> DetailedIntentAnalysis:
        """Synthesize all perspectives into a comprehensive analysis"""
        
        # Calculate overall confidence as weighted average
        total_confidence = 0
        total_weight = 0
        for perspective in perspectives.values():
            weight = perspective.estimated_effort  # Use effort as weight
            total_confidence += perspective.confidence_level * weight
            total_weight += weight
        
        overall_confidence = total_confidence / total_weight if total_weight > 0 else 0.5
        
        # Map string values to enums
        query_type = QueryType(basic_analysis.get("query_type", "analytical"))
        complexity_level = self._map_complexity(basic_analysis.get("complexity_level", "moderate"))
        urgency_level = self._map_urgency(basic_analysis.get("urgency_level", "medium"))
        
        return DetailedIntentAnalysis(
            primary_intent=basic_analysis.get("primary_intent", ""),
            secondary_intents=basic_analysis.get("secondary_intents", []),
            query_type=query_type,
            complexity_level=complexity_level,
            urgency_level=urgency_level,
            perspectives=perspectives,
            overall_confidence=overall_confidence,
            execution_priority=basic_analysis.get("execution_priority", 5),
            estimated_timeline=basic_analysis.get("estimated_timeline", "Unknown"),
            critical_dependencies=basic_analysis.get("critical_dependencies", [])
        )
    
    def _clean_json_response(self, response) -> str:
        """Clean and prepare JSON response text"""
        response_text = response.content if hasattr(response, 'content') else str(response)
        response_text = response_text.strip()
        
        if response_text.startswith('```json'):
            response_text = response_text[7:]
        if response_text.endswith('```'):
            response_text = response_text[:-3]
        
        return response_text.strip()
    
    def _map_complexity(self, complexity_str: str) -> QueryComplexity:
        """Map string complexity to enum"""
        mapping = {
            "simple": QueryComplexity.SIMPLE,
            "moderate": QueryComplexity.MODERATE,
            "complex": QueryComplexity.COMPLEX,
            "very_complex": QueryComplexity.VERY_COMPLEX
        }
        return mapping.get(complexity_str, QueryComplexity.MODERATE)
    
    def _map_urgency(self, urgency_str: str) -> UrgencyLevel:
        """Map string urgency to enum"""
        mapping = {
            "low": UrgencyLevel.LOW,
            "medium": UrgencyLevel.MEDIUM,
            "high": UrgencyLevel.HIGH,
            "critical": UrgencyLevel.CRITICAL
        }
        return mapping.get(urgency_str, UrgencyLevel.MEDIUM)
    
    def _create_fallback_perspective_analysis(self, perspective: AnalysisPerspective) -> PerspectiveAnalysis:
        """Create fallback analysis for failed perspective"""
        return PerspectiveAnalysis(
            perspective=perspective,
            primary_concerns=["Analysis unavailable"],
            methodology_suggestions=["Standard approach"],
            potential_challenges=["Unknown challenges"],
            success_criteria=["Basic completion"],
            estimated_effort=0.5,
            confidence_level=0.3
        )
    
    def _create_fallback_analysis(self, query: str) -> DetailedIntentAnalysis:
        """Create fallback analysis when everything fails"""
        fallback_perspectives = {}
        for perspective_type in AnalysisPerspective:
            fallback_perspectives[perspective_type] = self._create_fallback_perspective_analysis(perspective_type)
        
        return DetailedIntentAnalysis(
            primary_intent="General analysis request",
            secondary_intents=[],
            query_type=QueryType.ANALYTICAL,
            complexity_level=QueryComplexity.MODERATE,
            urgency_level=UrgencyLevel.MEDIUM,
            perspectives=fallback_perspectives,
            overall_confidence=0.3,
            execution_priority=5,
            estimated_timeline="Unknown",
            critical_dependencies=["Data access"]
        )
    
    def get_perspective_summary(self, analysis: DetailedIntentAnalysis) -> str:
        """Generate human-readable summary of multi-perspective analysis"""
        
        summary = f"""
        🔍 Multi-Perspective Intent Analysis
        
        Primary Intent: {analysis.primary_intent}
        Query Type: {analysis.query_type.value}
        Complexity: {analysis.complexity_level.value}
        Urgency: {analysis.urgency_level.value}
        Overall Confidence: {analysis.overall_confidence:.2f}
        
        📊 Perspective Insights:
        """
        
        for perspective_type, perspective in analysis.perspectives.items():
            summary += f"""
        {perspective_type.value.title()}:
        • Primary Concerns: {', '.join(perspective.primary_concerns[:2])}
        • Suggested Methods: {', '.join(perspective.methodology_suggestions[:2])}
        • Confidence: {perspective.confidence_level:.2f}
        """
        
        summary += f"""
        🎯 Execution Details:
        • Priority: {analysis.execution_priority}/10
        • Timeline: {analysis.estimated_timeline}
        • Dependencies: {', '.join(analysis.critical_dependencies[:3])}
        """
        
        return summary.strip() 