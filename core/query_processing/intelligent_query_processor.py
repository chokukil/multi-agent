"""
Intelligent Query Processor for CherryAI LLM-First Enhancement

This module implements the core query processing logic that analyzes user queries
from multiple perspectives and enhances them for better AI agent coordination.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json

from core.llm_factory import create_llm_instance
from core.utils.logging import get_logger

logger = get_logger(__name__)

class QueryType(Enum):
    """Query types based on user intent"""
    ANALYTICAL = "analytical"           # 분석적 질문
    DIAGNOSTIC = "diagnostic"           # 진단적 질문  
    PREDICTIVE = "predictive"           # 예측적 질문
    PRESCRIPTIVE = "prescriptive"       # 처방적 질문
    EXPLORATORY = "exploratory"         # 탐색적 질문
    COMPARATIVE = "comparative"         # 비교적 질문
    OPTIMIZATION = "optimization"       # 최적화 질문

class DomainType(Enum):
    """Domain types for knowledge extraction"""
    MANUFACTURING = "manufacturing"     # 제조업
    HEALTHCARE = "healthcare"           # 의료
    FINANCE = "finance"                # 금융
    RETAIL = "retail"                  # 소매
    LOGISTICS = "logistics"            # 물류
    ENERGY = "energy"                  # 에너지
    GENERAL = "general"                # 범용
    MIXED = "mixed"                    # 혼합

class AnswerFormat(Enum):
    """Expected answer formats"""
    STRUCTURED_REPORT = "structured_report"     # 구조화된 보고서
    VISUAL_ANALYSIS = "visual_analysis"         # 시각적 분석
    TECHNICAL_SOLUTION = "technical_solution"   # 기술적 해법
    BUSINESS_INSIGHT = "business_insight"       # 비즈니스 인사이트
    COMPARATIVE_STUDY = "comparative_study"     # 비교 연구
    PREDICTIVE_MODEL = "predictive_model"       # 예측 모델
    PROCESS_OPTIMIZATION = "process_optimization" # 프로세스 최적화

@dataclass
class IntentAnalysis:
    """Multi-perspective intent analysis result"""
    primary_intent: str
    data_scientist_perspective: str
    domain_expert_perspective: str
    technical_implementer_perspective: str
    query_type: QueryType
    urgency_level: float  # 0.0 ~ 1.0
    complexity_score: float  # 0.0 ~ 1.0
    confidence_score: float  # 0.0 ~ 1.0
    
@dataclass
class DomainKnowledge:
    """Extracted domain knowledge"""
    domain_type: DomainType
    key_concepts: List[str]
    technical_terms: List[str]
    business_context: str
    required_expertise: List[str]
    relevant_methodologies: List[str]
    success_metrics: List[str]
    potential_challenges: List[str]
    
@dataclass
class AnswerStructure:
    """Predicted answer structure"""
    expected_format: AnswerFormat
    key_sections: List[str]
    required_visualizations: List[str]
    success_criteria: List[str]
    expected_deliverables: List[str]
    quality_checkpoints: List[str]
    
@dataclass
class EnhancedQuery:
    """Enhanced query with all analysis results"""
    original_query: str
    intent_analysis: IntentAnalysis
    domain_knowledge: DomainKnowledge
    answer_structure: AnswerStructure
    enhanced_queries: List[str]
    execution_strategy: str
    context_requirements: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    
class IntelligentQueryProcessor:
    """
    Intelligent Query Processor that enhances user queries through
    multi-perspective analysis and domain knowledge extraction.
    """
    
    def __init__(self):
        self.llm = create_llm_instance()
        self.intent_analyzer = None  # Will be implemented separately
        self.domain_extractor = None  # Will be implemented separately
        self.answer_predictor = None  # Will be implemented separately
        self.query_enhancer = None   # Will be implemented separately
        
        logger.info("🧠 IntelligentQueryProcessor initialized")
    
    async def process_query(self, user_query: str, data_context: Optional[Dict] = None) -> EnhancedQuery:
        """
        Process user query through comprehensive analysis pipeline
        
        Args:
            user_query: Original user query
            data_context: Available data context
            
        Returns:
            EnhancedQuery with comprehensive analysis
        """
        logger.info(f"🔍 Processing query: {user_query[:100]}...")
        
        try:
            # Step 1: Multi-perspective intent analysis
            intent_analysis = await self._analyze_intent_multi_perspective(user_query, data_context)
            
            # Step 2: Domain knowledge extraction
            domain_knowledge = await self._extract_domain_knowledge(user_query, intent_analysis)
            
            # Step 3: Answer structure prediction
            answer_structure = await self._predict_answer_structure(user_query, intent_analysis, domain_knowledge)
            
            # Step 4: Query enhancement
            enhanced_queries = await self._enhance_query_contextually(
                user_query, intent_analysis, domain_knowledge, answer_structure
            )
            
            # Step 5: Execution strategy determination
            execution_strategy = await self._determine_execution_strategy(
                intent_analysis, domain_knowledge, answer_structure
            )
            
            # Step 6: Context requirements identification
            context_requirements = await self._identify_context_requirements(
                intent_analysis, domain_knowledge, data_context
            )
            
            enhanced_query = EnhancedQuery(
                original_query=user_query,
                intent_analysis=intent_analysis,
                domain_knowledge=domain_knowledge,
                answer_structure=answer_structure,
                enhanced_queries=enhanced_queries,
                execution_strategy=execution_strategy,
                context_requirements=context_requirements
            )
            
            logger.info("✅ Query processing completed successfully")
            return enhanced_query
            
        except Exception as e:
            logger.error(f"❌ Query processing failed: {str(e)}")
            raise
    
    async def _analyze_intent_multi_perspective(self, query: str, data_context: Optional[Dict] = None) -> IntentAnalysis:
        """Analyze query intent from multiple perspectives"""
        
        prompt = f"""
        You are an AI analysis expert. Analyze the following user query from three different perspectives.
        
        Query: "{query}"
        Data Context: {json.dumps(data_context, indent=2) if data_context else "None"}
        
        Please provide analysis from:
        1. DATA SCIENTIST perspective (methodology, statistical approach, data requirements)
        2. DOMAIN EXPERT perspective (business context, industry knowledge, practical implications)
        3. TECHNICAL IMPLEMENTER perspective (implementation challenges, system requirements, technical feasibility)
        
        Also classify the query type and assess urgency, complexity, and confidence levels.
        
        IMPORTANT: Respond ONLY with valid JSON in this exact format:
        {{
            "primary_intent": "What the user fundamentally wants to achieve",
            "data_scientist_perspective": "Technical analysis approach and methodology",
            "domain_expert_perspective": "Business context and industry-specific insights",
            "technical_implementer_perspective": "Implementation considerations and technical challenges",
            "query_type": "analytical",
            "urgency_level": 0.7,
            "complexity_score": 0.8,
            "confidence_score": 0.9
        }}
        """
        
        try:
            response = await asyncio.to_thread(self.llm.invoke, prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Clean up response text
            response_text = response_text.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            
            logger.debug(f"Raw LLM response: {response_text}")
            
            # Parse JSON response
            analysis_data = json.loads(response_text)
            
            return IntentAnalysis(
                primary_intent=analysis_data.get("primary_intent", ""),
                data_scientist_perspective=analysis_data.get("data_scientist_perspective", ""),
                domain_expert_perspective=analysis_data.get("domain_expert_perspective", ""),
                technical_implementer_perspective=analysis_data.get("technical_implementer_perspective", ""),
                query_type=QueryType(analysis_data.get("query_type", "analytical")),
                urgency_level=float(analysis_data.get("urgency_level", 0.5)),
                complexity_score=float(analysis_data.get("complexity_score", 0.5)),
                confidence_score=float(analysis_data.get("confidence_score", 0.5))
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {str(e)}, Response: {response_text[:200]}")
            # Return default analysis
            return IntentAnalysis(
                primary_intent="General data analysis request",
                data_scientist_perspective="Requires exploratory data analysis",
                domain_expert_perspective="Domain context needs to be identified",
                technical_implementer_perspective="Standard data processing pipeline",
                query_type=QueryType.ANALYTICAL,
                urgency_level=0.5,
                complexity_score=0.5,
                confidence_score=0.3
            )
        except Exception as e:
            logger.error(f"Intent analysis failed: {str(e)}")
            # Return default analysis
            return IntentAnalysis(
                primary_intent="General data analysis request",
                data_scientist_perspective="Requires exploratory data analysis",
                domain_expert_perspective="Domain context needs to be identified",
                technical_implementer_perspective="Standard data processing pipeline",
                query_type=QueryType.ANALYTICAL,
                urgency_level=0.5,
                complexity_score=0.5,
                confidence_score=0.3
            )
    
    async def _extract_domain_knowledge(self, query: str, intent_analysis: IntentAnalysis) -> DomainKnowledge:
        """Extract domain-specific knowledge from query"""
        
        prompt = f"""
        You are a domain knowledge expert. Extract domain-specific knowledge from the following query and intent analysis.
        
        Query: "{query}"
        Primary Intent: {intent_analysis.primary_intent}
        Domain Expert Perspective: {intent_analysis.domain_expert_perspective}
        
        Please identify:
        1. The primary domain (manufacturing, healthcare, finance, etc.)
        2. Key domain concepts and terminology
        3. Business context and requirements
        4. Required expertise areas
        5. Relevant methodologies for this domain
        6. Success metrics specific to this domain
        7. Potential challenges in this domain
        
        IMPORTANT: Respond ONLY with valid JSON in this exact format:
        {{
            "domain_type": "manufacturing",
            "key_concepts": ["concept1", "concept2"],
            "technical_terms": ["term1", "term2"],
            "business_context": "Description of business context",
            "required_expertise": ["expertise1", "expertise2"],
            "relevant_methodologies": ["methodology1", "methodology2"],
            "success_metrics": ["metric1", "metric2"],
            "potential_challenges": ["challenge1", "challenge2"]
        }}
        """
        
        try:
            response = await asyncio.to_thread(self.llm.invoke, prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Clean up response text
            response_text = response_text.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            
            logger.debug(f"Raw LLM response: {response_text}")
            
            # Parse JSON response
            knowledge_data = json.loads(response_text)
            
            return DomainKnowledge(
                domain_type=DomainType(knowledge_data.get("domain_type", "general")),
                key_concepts=knowledge_data.get("key_concepts", []),
                technical_terms=knowledge_data.get("technical_terms", []),
                business_context=knowledge_data.get("business_context", ""),
                required_expertise=knowledge_data.get("required_expertise", []),
                relevant_methodologies=knowledge_data.get("relevant_methodologies", []),
                success_metrics=knowledge_data.get("success_metrics", []),
                potential_challenges=knowledge_data.get("potential_challenges", [])
            )
            
        except Exception as e:
            logger.error(f"Domain knowledge extraction failed: {str(e)}")
            # Return default domain knowledge
            return DomainKnowledge(
                domain_type=DomainType.GENERAL,
                key_concepts=["data analysis", "pattern recognition"],
                technical_terms=["statistics", "correlation", "regression"],
                business_context="General business intelligence and analytics",
                required_expertise=["data science", "statistics"],
                relevant_methodologies=["exploratory data analysis", "statistical modeling"],
                success_metrics=["accuracy", "insight quality"],
                potential_challenges=["data quality", "interpretation complexity"]
            )
    
    async def _predict_answer_structure(self, query: str, intent_analysis: IntentAnalysis, domain_knowledge: DomainKnowledge) -> AnswerStructure:
        """Predict the optimal answer structure"""
        
        prompt = f"""
        You are an expert in answer structure design. Based on the query and analysis, predict the optimal answer structure.
        
        Query: "{query}"
        Query Type: {intent_analysis.query_type.value}
        Primary Intent: {intent_analysis.primary_intent}
        Domain: {domain_knowledge.domain_type.value}
        Business Context: {domain_knowledge.business_context}
        
        Please determine:
        1. Expected answer format (structured_report, visual_analysis, technical_solution, etc.)
        2. Key sections the answer should include
        3. Required visualizations
        4. Success criteria for the answer
        5. Expected deliverables
        6. Quality checkpoints
        
        IMPORTANT: Respond ONLY with valid JSON in this exact format:
        {{
            "expected_format": "structured_report",
            "key_sections": ["section1", "section2"],
            "required_visualizations": ["viz1", "viz2"],
            "success_criteria": ["criteria1", "criteria2"],
            "expected_deliverables": ["deliverable1", "deliverable2"],
            "quality_checkpoints": ["checkpoint1", "checkpoint2"]
        }}
        """
        
        try:
            response = await asyncio.to_thread(self.llm.invoke, prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Clean up response text
            response_text = response_text.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            
            logger.debug(f"Raw LLM response: {response_text}")
            
            # Parse JSON response
            structure_data = json.loads(response_text)
            
            return AnswerStructure(
                expected_format=AnswerFormat(structure_data.get("expected_format", "structured_report")),
                key_sections=structure_data.get("key_sections", []),
                required_visualizations=structure_data.get("required_visualizations", []),
                success_criteria=structure_data.get("success_criteria", []),
                expected_deliverables=structure_data.get("expected_deliverables", []),
                quality_checkpoints=structure_data.get("quality_checkpoints", [])
            )
            
        except Exception as e:
            logger.error(f"Answer structure prediction failed: {str(e)}")
            # Return default structure
            return AnswerStructure(
                expected_format=AnswerFormat.STRUCTURED_REPORT,
                key_sections=["Executive Summary", "Analysis", "Findings", "Recommendations"],
                required_visualizations=["charts", "graphs"],
                success_criteria=["accuracy", "completeness"],
                expected_deliverables=["analysis report", "visualizations"],
                quality_checkpoints=["data validation", "result verification"]
            )
    
    async def _enhance_query_contextually(self, query: str, intent_analysis: IntentAnalysis, 
                                        domain_knowledge: DomainKnowledge, answer_structure: AnswerStructure) -> List[str]:
        """Generate enhanced queries with context"""
        
        prompt = f"""
        You are a query enhancement expert. Generate enhanced versions of the original query that incorporate the analysis insights.
        
        Original Query: "{query}"
        
        Analysis Context:
        - Primary Intent: {intent_analysis.primary_intent}
        - Domain: {domain_knowledge.domain_type.value}
        - Key Concepts: {', '.join(domain_knowledge.key_concepts)}
        - Required Expertise: {', '.join(domain_knowledge.required_expertise)}
        - Expected Format: {answer_structure.expected_format.value}
        
        Generate 3-5 enhanced queries that:
        1. Incorporate domain-specific terminology
        2. Clarify the analysis requirements
        3. Specify the expected deliverables
        4. Include relevant methodologies
        5. Address potential challenges
        
        IMPORTANT: Respond ONLY with valid JSON array of strings in this exact format:
        ["enhanced_query_1", "enhanced_query_2", "enhanced_query_3"]
        """
        
        try:
            response = await asyncio.to_thread(self.llm.invoke, prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Clean up response text
            response_text = response_text.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            
            logger.debug(f"Raw LLM response: {response_text}")
            
            # Parse JSON response
            enhanced_queries = json.loads(response_text)
            
            return enhanced_queries if isinstance(enhanced_queries, list) else [query]
            
        except Exception as e:
            logger.error(f"Query enhancement failed: {str(e)}")
            return [query]  # Return original query as fallback
    
    async def _determine_execution_strategy(self, intent_analysis: IntentAnalysis, 
                                          domain_knowledge: DomainKnowledge, 
                                          answer_structure: AnswerStructure) -> str:
        """Determine the optimal execution strategy"""
        
        complexity = intent_analysis.complexity_score
        domain_type = domain_knowledge.domain_type
        answer_format = answer_structure.expected_format
        
        # Strategy determination logic
        if complexity > 0.8:
            strategy = "complex_multi_agent_orchestration"
        elif complexity > 0.6:
            strategy = "structured_multi_agent_collaboration"
        elif complexity > 0.4:
            strategy = "sequential_agent_execution"
        else:
            strategy = "single_agent_analysis"
        
        # Adjust based on domain and format
        if domain_type in [DomainType.MANUFACTURING, DomainType.HEALTHCARE]:
            strategy = f"domain_aware_{strategy}"
        
        if answer_format in [AnswerFormat.PREDICTIVE_MODEL, AnswerFormat.PROCESS_OPTIMIZATION]:
            strategy = f"advanced_{strategy}"
        
        return strategy
    
    async def _identify_context_requirements(self, intent_analysis: IntentAnalysis, 
                                           domain_knowledge: DomainKnowledge, 
                                           data_context: Optional[Dict]) -> Dict[str, Any]:
        """Identify required context for execution"""
        
        requirements = {
            "data_requirements": domain_knowledge.technical_terms,
            "expertise_requirements": domain_knowledge.required_expertise,
            "methodology_requirements": domain_knowledge.relevant_methodologies,
            "quality_requirements": domain_knowledge.success_metrics,
            "domain_context": domain_knowledge.business_context,
            "urgency_level": intent_analysis.urgency_level,
            "complexity_score": intent_analysis.complexity_score,
            "available_data": data_context or {}
        }
        
        return requirements
    
    def get_query_summary(self, enhanced_query: EnhancedQuery) -> str:
        """Generate a human-readable summary of the query analysis"""
        
        summary = f"""
        🔍 Query Analysis Summary
        
        Original Query: {enhanced_query.original_query}
        
        📊 Intent Analysis:
        • Primary Intent: {enhanced_query.intent_analysis.primary_intent}
        • Query Type: {enhanced_query.intent_analysis.query_type.value}
        • Complexity: {enhanced_query.intent_analysis.complexity_score:.2f}
        • Urgency: {enhanced_query.intent_analysis.urgency_level:.2f}
        
        🎯 Domain Knowledge:
        • Domain: {enhanced_query.domain_knowledge.domain_type.value}
        • Key Concepts: {', '.join(enhanced_query.domain_knowledge.key_concepts[:3])}
        • Required Expertise: {', '.join(enhanced_query.domain_knowledge.required_expertise[:3])}
        
        📋 Expected Answer:
        • Format: {enhanced_query.answer_structure.expected_format.value}
        • Key Sections: {', '.join(enhanced_query.answer_structure.key_sections[:3])}
        
        🚀 Execution Strategy: {enhanced_query.execution_strategy}
        """
        
        return summary.strip() 