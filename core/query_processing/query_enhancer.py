"""
Contextual Query Enhancer for CherryAI

This module enhances user queries by incorporating insights from intent analysis,
domain knowledge extraction, and answer structure prediction to create optimized
queries for better AI agent coordination.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum

from core.llm_factory import create_llm_instance
from core.utils.logging import get_logger
from .intent_analyzer import DetailedIntentAnalysis
from .domain_extractor import EnhancedDomainKnowledge
from .answer_predictor import PredictedAnswerStructure

logger = get_logger(__name__)

class EnhancementType(Enum):
    """Types of query enhancements"""
    TERMINOLOGY_ENRICHMENT = "terminology_enrichment"      # Add domain-specific terms
    CONTEXT_CLARIFICATION = "context_clarification"       # Clarify implicit context
    METHODOLOGY_SPECIFICATION = "methodology_specification" # Specify analytical methods
    DELIVERABLE_DEFINITION = "deliverable_definition"     # Define expected outputs
    CONSTRAINT_ADDITION = "constraint_addition"           # Add relevant constraints
    STAKEHOLDER_ALIGNMENT = "stakeholder_alignment"       # Align with stakeholder needs
    SUCCESS_CRITERIA = "success_criteria"                 # Add success metrics
    RISK_CONSIDERATION = "risk_consideration"             # Include risk factors

class QueryEnhancementStrategy(Enum):
    """Strategies for query enhancement"""
    MINIMAL_ENHANCEMENT = "minimal_enhancement"           # Light enhancement
    MODERATE_ENHANCEMENT = "moderate_enhancement"         # Balanced enhancement
    COMPREHENSIVE_ENHANCEMENT = "comprehensive_enhancement" # Full enhancement
    DOMAIN_FOCUSED = "domain_focused"                     # Domain-specific focus
    TECHNICAL_FOCUSED = "technical_focused"               # Technical implementation focus
    BUSINESS_FOCUSED = "business_focused"                 # Business outcome focus

@dataclass
class EnhancementRule:
    """Rule for query enhancement"""
    enhancement_type: EnhancementType
    condition: str
    template: str
    priority: int  # 1-10
    applicability_score: float

@dataclass
class EnhancedQuery:
    """Enhanced query with metadata"""
    original_query: str
    enhanced_query: str
    enhancement_types: List[EnhancementType]
    enhancement_reasoning: str
    confidence_score: float
    estimated_improvement: float
    enhancement_metadata: Dict[str, any]

@dataclass
class QueryVariation:
    """A variation of the enhanced query"""
    variation_query: str
    variation_type: str
    target_use_case: str
    relative_priority: int
    expected_outcomes: List[str]

@dataclass
class ComprehensiveQueryEnhancement:
    """Complete query enhancement package"""
    primary_enhanced_query: EnhancedQuery
    query_variations: List[QueryVariation]
    enhancement_strategy: QueryEnhancementStrategy
    optimization_confidence: float
    usage_recommendations: Dict[str, str]

class ContextualQueryEnhancer:
    """
    Advanced query enhancer that leverages comprehensive analysis results
    to create optimized queries for better AI agent coordination.
    """
    
    def __init__(self):
        self.llm = create_llm_instance()
        
        # Initialize enhancement rules and strategies
        self.enhancement_rules = self._initialize_enhancement_rules()
        self.strategy_templates = self._initialize_strategy_templates()
        self.domain_vocabularies = self._initialize_domain_vocabularies()
        
        logger.info("🚀 ContextualQueryEnhancer initialized")
    
    async def enhance_query_comprehensively(self, 
                                          original_query: str,
                                          intent_analysis: DetailedIntentAnalysis,
                                          domain_knowledge: EnhancedDomainKnowledge,
                                          answer_structure: PredictedAnswerStructure,
                                          enhancement_strategy: Optional[QueryEnhancementStrategy] = None) -> ComprehensiveQueryEnhancement:
        """
        Enhance query comprehensively using all available analysis results
        
        Args:
            original_query: Original user query
            intent_analysis: Detailed intent analysis results
            domain_knowledge: Comprehensive domain knowledge
            answer_structure: Predicted answer structure
            enhancement_strategy: Optional enhancement strategy
            
        Returns:
            ComprehensiveQueryEnhancement with optimized queries
        """
        logger.info(f"🔧 Enhancing query comprehensively: {original_query[:100]}...")
        
        try:
            # Step 1: Determine enhancement strategy
            if not enhancement_strategy:
                enhancement_strategy = await self._determine_enhancement_strategy(
                    original_query, intent_analysis, domain_knowledge, answer_structure
                )
            
            # Step 2: Apply terminology enrichment
            terminology_enhanced = await self._apply_terminology_enrichment(
                original_query, domain_knowledge, enhancement_strategy
            )
            
            # Step 3: Add context clarification
            context_clarified = await self._apply_context_clarification(
                terminology_enhanced, intent_analysis, domain_knowledge
            )
            
            # Step 4: Specify methodologies
            methodology_specified = await self._apply_methodology_specification(
                context_clarified, domain_knowledge, answer_structure
            )
            
            # Step 5: Define deliverables
            deliverable_defined = await self._apply_deliverable_definition(
                methodology_specified, answer_structure, intent_analysis
            )
            
            # Step 6: Add constraints and success criteria
            fully_enhanced = await self._apply_constraints_and_criteria(
                deliverable_defined, domain_knowledge, intent_analysis
            )
            
            # Step 7: Create primary enhanced query
            primary_enhanced = await self._create_primary_enhanced_query(
                original_query, fully_enhanced, intent_analysis, domain_knowledge, enhancement_strategy
            )
            
            # Step 8: Generate query variations
            variations = await self._generate_query_variations(
                primary_enhanced, intent_analysis, domain_knowledge, answer_structure
            )
            
            # Step 9: Calculate optimization confidence
            optimization_confidence = self._calculate_optimization_confidence(
                primary_enhanced, intent_analysis, domain_knowledge
            )
            
            # Step 10: Generate usage recommendations
            usage_recommendations = await self._generate_usage_recommendations(
                primary_enhanced, variations, enhancement_strategy
            )
            
            return ComprehensiveQueryEnhancement(
                primary_enhanced_query=primary_enhanced,
                query_variations=variations,
                enhancement_strategy=enhancement_strategy,
                optimization_confidence=optimization_confidence,
                usage_recommendations=usage_recommendations
            )
            
        except Exception as e:
            logger.error(f"❌ Comprehensive query enhancement failed: {str(e)}")
            return self._create_fallback_enhancement(original_query)
    
    async def _determine_enhancement_strategy(self, 
                                            original_query: str,
                                            intent_analysis: DetailedIntentAnalysis,
                                            domain_knowledge: EnhancedDomainKnowledge,
                                            answer_structure: PredictedAnswerStructure) -> QueryEnhancementStrategy:
        """Determine the optimal enhancement strategy"""
        
        prompt = f"""
        You are a query optimization expert. Determine the optimal enhancement strategy for this query.
        
        Original Query: "{original_query}"
        
        Analysis Context:
        • Query Complexity: {intent_analysis.complexity_level.value}
        • Domain: {domain_knowledge.taxonomy.primary_domain.value}
        • Target Audience: {answer_structure.primary_template.target_audience}
        • Overall Confidence: {intent_analysis.overall_confidence:.2f}
        
        Available enhancement strategies:
        • minimal_enhancement: Light touch, preserve user intent
        • moderate_enhancement: Balanced enhancement with clarity improvements
        • comprehensive_enhancement: Full enhancement with all context
        • domain_focused: Emphasize domain-specific aspects
        • technical_focused: Focus on technical implementation
        • business_focused: Focus on business outcomes
        
        Consider:
        1. How complex is the original query?
        2. How much domain context is needed?
        3. What does the target audience expect?
        4. What level of enhancement adds value without over-engineering?
        
        IMPORTANT: Respond ONLY with valid JSON:
        {{
            "strategy": "moderate_enhancement",
            "reasoning": "Explanation for strategy choice",
            "confidence": 0.85
        }}
        """
        
        try:
            response = await asyncio.to_thread(self.llm.invoke, prompt)
            response_text = self._clean_json_response(response)
            data = json.loads(response_text)
            
            return QueryEnhancementStrategy(data.get("strategy", "moderate_enhancement"))
            
        except Exception as e:
            logger.error(f"Enhancement strategy determination failed: {str(e)}")
            return QueryEnhancementStrategy.MODERATE_ENHANCEMENT
    
    async def _apply_terminology_enrichment(self, 
                                          query: str, 
                                          domain_knowledge: EnhancedDomainKnowledge,
                                          strategy: QueryEnhancementStrategy) -> str:
        """Apply domain-specific terminology enrichment"""
        
        if strategy == QueryEnhancementStrategy.MINIMAL_ENHANCEMENT:
            return query  # Skip for minimal enhancement
        
        prompt = f"""
        You are a domain terminology expert. Enhance this query with appropriate domain-specific terminology.
        
        Original Query: "{query}"
        
        Domain Context:
        • Primary Domain: {domain_knowledge.taxonomy.primary_domain.value}
        • Technical Area: {domain_knowledge.taxonomy.technical_area}
        • Key Concepts: {', '.join(list(domain_knowledge.key_concepts.keys())[:5])}
        • Technical Terms: {', '.join(list(domain_knowledge.technical_terms.keys())[:5])}
        
        Guidelines:
        1. Replace generic terms with domain-specific terminology
        2. Add clarifying technical terms where helpful
        3. Maintain the original intent and structure
        4. Keep the query natural and readable
        
        Enhanced query (return only the enhanced query text):
        """
        
        try:
            response = await asyncio.to_thread(self.llm.invoke, prompt)
            enhanced_text = response.content if hasattr(response, 'content') else str(response)
            return enhanced_text.strip()
            
        except Exception as e:
            logger.error(f"Terminology enrichment failed: {str(e)}")
            return query
    
    async def _apply_context_clarification(self, 
                                         query: str,
                                         intent_analysis: DetailedIntentAnalysis,
                                         domain_knowledge: EnhancedDomainKnowledge) -> str:
        """Apply context clarification to make implicit context explicit"""
        
        prompt = f"""
        You are a context clarification expert. Enhance this query by making implicit context explicit.
        
        Current Query: "{query}"
        
        Context to Clarify:
        • Primary Intent: {intent_analysis.primary_intent}
        • Secondary Intents: {', '.join(intent_analysis.secondary_intents)}
        • Business Context: {domain_knowledge.business_context}
        • Key Stakeholders: {', '.join(domain_knowledge.stakeholder_map.get('primary', []))}
        
        Guidelines:
        1. Make implicit assumptions explicit
        2. Clarify the business context
        3. Specify the scope and boundaries
        4. Add relevant constraints
        
        Context-clarified query (return only the enhanced query text):
        """
        
        try:
            response = await asyncio.to_thread(self.llm.invoke, prompt)
            enhanced_text = response.content if hasattr(response, 'content') else str(response)
            return enhanced_text.strip()
            
        except Exception as e:
            logger.error(f"Context clarification failed: {str(e)}")
            return query
    
    async def _apply_methodology_specification(self, 
                                             query: str,
                                             domain_knowledge: EnhancedDomainKnowledge,
                                             answer_structure: PredictedAnswerStructure) -> str:
        """Apply methodology specification to guide analytical approach"""
        
        prompt = f"""
        You are a methodology expert. Enhance this query by specifying analytical methodologies.
        
        Current Query: "{query}"
        
        Methodology Context:
        • Standard Methodologies: {', '.join(domain_knowledge.methodology_map.standard_methodologies[:3])}
        • Best Practices: {', '.join(domain_knowledge.methodology_map.best_practices[:3])}
        • Required Visualizations: {', '.join([v.value for v in answer_structure.primary_template.required_visualizations[:3]])}
        
        Guidelines:
        1. Suggest appropriate analytical methodologies
        2. Specify data analysis approaches
        3. Mention relevant statistical techniques
        4. Include visualization requirements
        
        Methodology-enhanced query (return only the enhanced query text):
        """
        
        try:
            response = await asyncio.to_thread(self.llm.invoke, prompt)
            enhanced_text = response.content if hasattr(response, 'content') else str(response)
            return enhanced_text.strip()
            
        except Exception as e:
            logger.error(f"Methodology specification failed: {str(e)}")
            return query
    
    async def _apply_deliverable_definition(self, 
                                          query: str,
                                          answer_structure: PredictedAnswerStructure,
                                          intent_analysis: DetailedIntentAnalysis) -> str:
        """Apply deliverable definition to specify expected outputs"""
        
        template = answer_structure.primary_template
        
        prompt = f"""
        You are a deliverable specification expert. Enhance this query by defining expected deliverables.
        
        Current Query: "{query}"
        
        Expected Deliverables:
        • Format: {template.format_type.value}
        • Key Sections: {', '.join([s.title for s in template.sections[:5]])}
        • Target Audience: {template.target_audience}
        • Completion Time: {template.estimated_completion_time}
        
        Guidelines:
        1. Specify the expected output format
        2. Define key deliverable components
        3. Clarify the target audience
        4. Set realistic expectations for completeness
        
        Deliverable-enhanced query (return only the enhanced query text):
        """
        
        try:
            response = await asyncio.to_thread(self.llm.invoke, prompt)
            enhanced_text = response.content if hasattr(response, 'content') else str(response)
            return enhanced_text.strip()
            
        except Exception as e:
            logger.error(f"Deliverable definition failed: {str(e)}")
            return query
    
    async def _apply_constraints_and_criteria(self, 
                                            query: str,
                                            domain_knowledge: EnhancedDomainKnowledge,
                                            intent_analysis: DetailedIntentAnalysis) -> str:
        """Apply constraints and success criteria"""
        
        prompt = f"""
        You are a requirements expert. Enhance this query by adding relevant constraints and success criteria.
        
        Current Query: "{query}"
        
        Context:
        • Success Metrics: {', '.join(domain_knowledge.success_metrics[:3])}
        • Potential Risks: {', '.join(domain_knowledge.risk_assessment.technical_risks[:2])}
        • Timeline: {intent_analysis.estimated_timeline}
        • Priority: {intent_analysis.execution_priority}/10
        
        Guidelines:
        1. Add relevant constraints (time, quality, scope)
        2. Include success criteria
        3. Mention risk considerations
        4. Specify quality requirements
        
        Fully enhanced query (return only the enhanced query text):
        """
        
        try:
            response = await asyncio.to_thread(self.llm.invoke, prompt)
            enhanced_text = response.content if hasattr(response, 'content') else str(response)
            return enhanced_text.strip()
            
        except Exception as e:
            logger.error(f"Constraints and criteria application failed: {str(e)}")
            return query
    
    async def _create_primary_enhanced_query(self, 
                                           original_query: str,
                                           enhanced_query: str,
                                           intent_analysis: DetailedIntentAnalysis,
                                           domain_knowledge: EnhancedDomainKnowledge,
                                           strategy: QueryEnhancementStrategy) -> EnhancedQuery:
        """Create the primary enhanced query object"""
        
        # Determine enhancement types applied
        enhancement_types = []
        if enhanced_query != original_query:
            enhancement_types = [
                EnhancementType.TERMINOLOGY_ENRICHMENT,
                EnhancementType.CONTEXT_CLARIFICATION,
                EnhancementType.METHODOLOGY_SPECIFICATION,
                EnhancementType.DELIVERABLE_DEFINITION,
                EnhancementType.CONSTRAINT_ADDITION
            ]
        
        # Generate enhancement reasoning
        reasoning = await self._generate_enhancement_reasoning(
            original_query, enhanced_query, intent_analysis, domain_knowledge, strategy
        )
        
        # Calculate confidence and improvement scores
        confidence_score = self._calculate_enhancement_confidence(
            original_query, enhanced_query, intent_analysis, domain_knowledge
        )
        
        estimated_improvement = self._calculate_estimated_improvement(
            original_query, enhanced_query, enhancement_types
        )
        
        # Create metadata
        metadata = {
            "original_length": len(original_query),
            "enhanced_length": len(enhanced_query),
            "enhancement_ratio": len(enhanced_query) / len(original_query),
            "strategy_applied": strategy.value,
            "domain_confidence": domain_knowledge.extraction_confidence,
            "intent_confidence": intent_analysis.overall_confidence
        }
        
        return EnhancedQuery(
            original_query=original_query,
            enhanced_query=enhanced_query,
            enhancement_types=enhancement_types,
            enhancement_reasoning=reasoning,
            confidence_score=confidence_score,
            estimated_improvement=estimated_improvement,
            enhancement_metadata=metadata
        )
    
    async def _generate_query_variations(self, 
                                       primary_enhanced: EnhancedQuery,
                                       intent_analysis: DetailedIntentAnalysis,
                                       domain_knowledge: EnhancedDomainKnowledge,
                                       answer_structure: PredictedAnswerStructure) -> List[QueryVariation]:
        """Generate query variations for different use cases"""
        
        variations = []
        
        # Executive summary variation
        if answer_structure.primary_template.target_audience != "executive":
            exec_variation = await self._create_executive_variation(primary_enhanced, domain_knowledge)
            variations.append(exec_variation)
        
        # Technical detail variation
        if answer_structure.primary_template.target_audience != "technical":
            tech_variation = await self._create_technical_variation(primary_enhanced, domain_knowledge)
            variations.append(tech_variation)
        
        # Quick analysis variation
        if intent_analysis.complexity_level.value in ["complex", "very_complex"]:
            quick_variation = await self._create_quick_analysis_variation(primary_enhanced, intent_analysis)
            variations.append(quick_variation)
        
        return variations
    
    async def _create_executive_variation(self, primary_enhanced: EnhancedQuery, domain_knowledge: EnhancedDomainKnowledge) -> QueryVariation:
        """Create executive-focused variation"""
        
        prompt = f"""
        Create an executive-focused variation of this enhanced query.
        
        Enhanced Query: "{primary_enhanced.enhanced_query}"
        Business Context: {domain_knowledge.business_context}
        
        Focus on:
        1. Business impact and value
        2. High-level outcomes
        3. Strategic implications
        4. Executive decision support
        
        Executive variation (return only the query text):
        """
        
        try:
            response = await asyncio.to_thread(self.llm.invoke, prompt)
            variation_text = response.content if hasattr(response, 'content') else str(response)
            
            return QueryVariation(
                variation_query=variation_text.strip(),
                variation_type="executive_focused",
                target_use_case="Executive briefing and strategic decision making",
                relative_priority=8,
                expected_outcomes=["Business impact assessment", "Strategic recommendations", "Executive summary"]
            )
            
        except Exception as e:
            logger.error(f"Executive variation creation failed: {str(e)}")
            return QueryVariation(
                variation_query=primary_enhanced.enhanced_query,
                variation_type="executive_focused",
                target_use_case="Executive briefing",
                relative_priority=5,
                expected_outcomes=["Summary report"]
            )
    
    async def _create_technical_variation(self, primary_enhanced: EnhancedQuery, domain_knowledge: EnhancedDomainKnowledge) -> QueryVariation:
        """Create technical-focused variation"""
        
        prompt = f"""
        Create a technical-focused variation of this enhanced query.
        
        Enhanced Query: "{primary_enhanced.enhanced_query}"
        Technical Area: {domain_knowledge.taxonomy.technical_area}
        Standard Methodologies: {', '.join(domain_knowledge.methodology_map.standard_methodologies[:3])}
        
        Focus on:
        1. Technical implementation details
        2. Methodological rigor
        3. Technical specifications
        4. Implementation guidance
        
        Technical variation (return only the query text):
        """
        
        try:
            response = await asyncio.to_thread(self.llm.invoke, prompt)
            variation_text = response.content if hasattr(response, 'content') else str(response)
            
            return QueryVariation(
                variation_query=variation_text.strip(),
                variation_type="technical_focused",
                target_use_case="Technical implementation and detailed analysis",
                relative_priority=7,
                expected_outcomes=["Technical specifications", "Implementation plan", "Methodology details"]
            )
            
        except Exception as e:
            logger.error(f"Technical variation creation failed: {str(e)}")
            return QueryVariation(
                variation_query=primary_enhanced.enhanced_query,
                variation_type="technical_focused",
                target_use_case="Technical analysis",
                relative_priority=5,
                expected_outcomes=["Technical report"]
            )
    
    async def _create_quick_analysis_variation(self, primary_enhanced: EnhancedQuery, intent_analysis: DetailedIntentAnalysis) -> QueryVariation:
        """Create quick analysis variation for complex queries"""
        
        prompt = f"""
        Create a quick analysis variation of this complex query.
        
        Enhanced Query: "{primary_enhanced.enhanced_query}"
        Primary Intent: {intent_analysis.primary_intent}
        
        Focus on:
        1. Quick wins and immediate insights
        2. Essential findings only
        3. Rapid turnaround
        4. Key recommendations
        
        Quick analysis variation (return only the query text):
        """
        
        try:
            response = await asyncio.to_thread(self.llm.invoke, prompt)
            variation_text = response.content if hasattr(response, 'content') else str(response)
            
            return QueryVariation(
                variation_query=variation_text.strip(),
                variation_type="quick_analysis",
                target_use_case="Rapid insights and immediate action items",
                relative_priority=9,
                expected_outcomes=["Quick insights", "Immediate actions", "Key findings"]
            )
            
        except Exception as e:
            logger.error(f"Quick analysis variation creation failed: {str(e)}")
            return QueryVariation(
                variation_query=primary_enhanced.enhanced_query,
                variation_type="quick_analysis",
                target_use_case="Quick analysis",
                relative_priority=5,
                expected_outcomes=["Quick summary"]
            )
    
    async def _generate_enhancement_reasoning(self, 
                                            original_query: str,
                                            enhanced_query: str,
                                            intent_analysis: DetailedIntentAnalysis,
                                            domain_knowledge: EnhancedDomainKnowledge,
                                            strategy: QueryEnhancementStrategy) -> str:
        """Generate reasoning for the enhancement"""
        
        reasoning_parts = [
            f"Applied {strategy.value} strategy to enhance query clarity and specificity",
            f"Incorporated {domain_knowledge.taxonomy.primary_domain.value} domain terminology and context",
            f"Added methodology specifications based on {domain_knowledge.taxonomy.technical_area}",
            f"Defined deliverables aligned with {intent_analysis.query_type.value} query type",
            f"Included success criteria relevant to {', '.join(domain_knowledge.success_metrics[:2])}"
        ]
        
        return ". ".join(reasoning_parts) + "."
    
    def _calculate_enhancement_confidence(self, 
                                        original_query: str,
                                        enhanced_query: str,
                                        intent_analysis: DetailedIntentAnalysis,
                                        domain_knowledge: EnhancedDomainKnowledge) -> float:
        """Calculate confidence in the enhancement"""
        
        base_confidence = (intent_analysis.overall_confidence + domain_knowledge.extraction_confidence) / 2
        
        # Adjust based on enhancement quality
        length_ratio = len(enhanced_query) / len(original_query)
        length_score = min(length_ratio / 2.0, 1.0) if length_ratio > 1.0 else 0.5
        
        # Adjust based on domain specificity
        domain_score = domain_knowledge.taxonomy.confidence_score
        
        overall_confidence = (base_confidence * 0.6 + length_score * 0.2 + domain_score * 0.2)
        
        return min(overall_confidence, 1.0)
    
    def _calculate_estimated_improvement(self, 
                                       original_query: str,
                                       enhanced_query: str,
                                       enhancement_types: List[EnhancementType]) -> float:
        """Calculate estimated improvement from enhancement"""
        
        base_improvement = len(enhancement_types) * 0.1  # 10% per enhancement type
        length_improvement = min((len(enhanced_query) - len(original_query)) / len(original_query), 0.3)
        
        return min(base_improvement + length_improvement, 0.8)  # Cap at 80% improvement
    
    def _calculate_optimization_confidence(self, 
                                         enhanced_query: EnhancedQuery,
                                         intent_analysis: DetailedIntentAnalysis,
                                         domain_knowledge: EnhancedDomainKnowledge) -> float:
        """Calculate overall optimization confidence"""
        
        enhancement_confidence = enhanced_query.confidence_score
        improvement_score = enhanced_query.estimated_improvement
        context_alignment = (intent_analysis.overall_confidence + domain_knowledge.extraction_confidence) / 2
        
        return (enhancement_confidence * 0.4 + improvement_score * 0.3 + context_alignment * 0.3)
    
    async def _generate_usage_recommendations(self, 
                                            enhanced_query: EnhancedQuery,
                                            variations: List[QueryVariation],
                                            strategy: QueryEnhancementStrategy) -> Dict[str, str]:
        """Generate usage recommendations for the enhanced queries"""
        
        recommendations = {
            "primary_use": f"Use the primary enhanced query for {strategy.value} analysis",
            "optimization_level": f"Enhancement provides {enhanced_query.estimated_improvement:.0%} estimated improvement",
            "confidence_level": f"Enhancement confidence: {enhanced_query.confidence_score:.0%}"
        }
        
        if variations:
            recommendations["variations"] = f"Consider {len(variations)} variations for different stakeholder needs"
            
            # Add specific variation recommendations
            for variation in variations:
                recommendations[f"{variation.variation_type}_use"] = variation.target_use_case
        
        return recommendations
    
    def _clean_json_response(self, response) -> str:
        """Clean and prepare JSON response text"""
        response_text = response.content if hasattr(response, 'content') else str(response)
        response_text = response_text.strip()
        
        if response_text.startswith('```json'):
            response_text = response_text[7:]
        if response_text.endswith('```'):
            response_text = response_text[:-3]
        
        return response_text.strip()
    
    def _initialize_enhancement_rules(self) -> List[EnhancementRule]:
        """Initialize enhancement rules"""
        return [
            EnhancementRule(
                enhancement_type=EnhancementType.TERMINOLOGY_ENRICHMENT,
                condition="domain_confidence > 0.7",
                template="Add domain-specific terminology",
                priority=8,
                applicability_score=0.8
            ),
            EnhancementRule(
                enhancement_type=EnhancementType.METHODOLOGY_SPECIFICATION,
                condition="technical_query",
                template="Specify analytical methodologies",
                priority=7,
                applicability_score=0.7
            )
        ]
    
    def _initialize_strategy_templates(self) -> Dict:
        """Initialize strategy templates"""
        return {
            QueryEnhancementStrategy.MINIMAL_ENHANCEMENT: "Light enhancement preserving user intent",
            QueryEnhancementStrategy.COMPREHENSIVE_ENHANCEMENT: "Full context enhancement with all available insights"
        }
    
    def _initialize_domain_vocabularies(self) -> Dict:
        """Initialize domain-specific vocabularies"""
        return {
            "manufacturing": ["process control", "quality assurance", "yield optimization"],
            "healthcare": ["clinical analysis", "patient outcomes", "treatment efficacy"]
        }
    
    def _create_fallback_enhancement(self, original_query: str) -> ComprehensiveQueryEnhancement:
        """Create fallback enhancement when processing fails"""
        
        fallback_enhanced = EnhancedQuery(
            original_query=original_query,
            enhanced_query=original_query,
            enhancement_types=[],
            enhancement_reasoning="Fallback due to enhancement processing failure",
            confidence_score=0.3,
            estimated_improvement=0.0,
            enhancement_metadata={"fallback": True}
        )
        
        return ComprehensiveQueryEnhancement(
            primary_enhanced_query=fallback_enhanced,
            query_variations=[],
            enhancement_strategy=QueryEnhancementStrategy.MINIMAL_ENHANCEMENT,
            optimization_confidence=0.3,
            usage_recommendations={"primary_use": "Use original query as-is"}
        )
    
    def get_enhancement_summary(self, enhancement: ComprehensiveQueryEnhancement) -> str:
        """Generate human-readable summary of enhancement"""
        
        enhanced = enhancement.primary_enhanced_query
        
        summary = f"""
        🔧 Query Enhancement Summary
        
        📝 Enhancement Results:
        • Strategy: {enhancement.enhancement_strategy.value}
        • Enhancement Types: {len(enhanced.enhancement_types)}
        • Estimated Improvement: {enhanced.estimated_improvement:.0%}
        • Enhancement Confidence: {enhanced.confidence_score:.2f}
        
        📏 Query Metrics:
        • Original Length: {enhanced.enhancement_metadata.get('original_length', 0)} chars
        • Enhanced Length: {enhanced.enhancement_metadata.get('enhanced_length', 0)} chars
        • Enhancement Ratio: {enhanced.enhancement_metadata.get('enhancement_ratio', 1.0):.1f}x
        
        🔄 Available Variations: {len(enhancement.query_variations)}
        {chr(10).join([f"• {v.variation_type}: {v.target_use_case}" for v in enhancement.query_variations])}
        
        💡 Enhancement Reasoning: {enhanced.enhancement_reasoning}
        
        ✅ Overall Optimization Confidence: {enhancement.optimization_confidence:.2f}
        
        📋 Usage Recommendations:
        {chr(10).join([f"• {k}: {v}" for k, v in enhancement.usage_recommendations.items()])}
        """
        
        return summary.strip() 