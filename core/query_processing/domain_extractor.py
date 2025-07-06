"""
Domain Knowledge Extractor for CherryAI

This module automatically extracts and structures domain-specific knowledge
from user queries to enable more intelligent analysis and processing.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum

from core.llm_factory import create_llm_instance
from core.utils.logging import get_logger
from .intelligent_query_processor import DomainType, DomainKnowledge

logger = get_logger(__name__)

class KnowledgeConfidence(Enum):
    """Confidence levels for extracted knowledge"""
    LOW = "low"             # 0.0 - 0.4
    MEDIUM = "medium"       # 0.4 - 0.7
    HIGH = "high"           # 0.7 - 0.9
    VERY_HIGH = "very_high" # 0.9 - 1.0

class KnowledgeSource(Enum):
    """Sources of domain knowledge"""
    EXPLICIT_MENTION = "explicit_mention"      # Directly mentioned in query
    CONTEXTUAL_INFERENCE = "contextual_inference"  # Inferred from context
    DOMAIN_TAXONOMY = "domain_taxonomy"        # From domain classification
    TECHNICAL_PATTERNS = "technical_patterns"  # From technical language patterns
    BUSINESS_CONTEXT = "business_context"      # From business implications

@dataclass
class KnowledgeItem:
    """Individual piece of domain knowledge"""
    item: str
    confidence: KnowledgeConfidence
    source: KnowledgeSource
    explanation: str
    related_items: List[str] = field(default_factory=list)

@dataclass
class DomainTaxonomy:
    """Structured domain taxonomy"""
    primary_domain: DomainType
    sub_domains: List[str]
    industry_sector: str
    business_function: str
    technical_area: str
    confidence_score: float

@dataclass
class MethodologyMap:
    """Domain-specific methodology mapping"""
    standard_methodologies: List[str]
    best_practices: List[str]
    tools_and_technologies: List[str]
    quality_standards: List[str]
    compliance_requirements: List[str]

@dataclass
class RiskAssessment:
    """Domain-specific risk assessment"""
    technical_risks: List[str]
    business_risks: List[str]
    operational_risks: List[str]
    compliance_risks: List[str]
    mitigation_strategies: List[str]

@dataclass
class EnhancedDomainKnowledge:
    """Comprehensive domain knowledge structure"""
    taxonomy: DomainTaxonomy
    key_concepts: Dict[str, KnowledgeItem]
    technical_terms: Dict[str, KnowledgeItem]
    methodology_map: MethodologyMap
    risk_assessment: RiskAssessment
    success_metrics: List[str]
    stakeholder_map: Dict[str, List[str]]
    business_context: str
    extraction_confidence: float

class DomainKnowledgeExtractor:
    """
    Advanced domain knowledge extractor that automatically identifies and
    structures domain-specific knowledge from user queries.
    """
    
    def __init__(self):
        self.llm = create_llm_instance()
        
        # Pre-loaded domain patterns and taxonomies
        self.domain_patterns = self._initialize_domain_patterns()
        self.methodology_database = self._initialize_methodology_database()
        
        logger.info("🧠 DomainKnowledgeExtractor initialized")
    
    async def extract_comprehensive_domain_knowledge(self, query: str, intent_analysis: Optional[Dict] = None, data_context: Optional[Dict] = None) -> EnhancedDomainKnowledge:
        """
        Extract comprehensive domain knowledge from query
        
        Args:
            query: User query to analyze
            intent_analysis: Optional intent analysis results
            data_context: Optional data context
            
        Returns:
            EnhancedDomainKnowledge with structured domain insights
        """
        logger.info(f"🔍 Extracting comprehensive domain knowledge for: {query[:100]}...")
        
        try:
            # Step 1: Domain taxonomy classification
            taxonomy = await self._classify_domain_taxonomy(query, intent_analysis, data_context)
            
            # Step 2: Extract key concepts and technical terms
            concepts_terms = await self._extract_concepts_and_terms(query, taxonomy)
            
            # Step 3: Map domain methodologies
            methodology_map = await self._map_domain_methodologies(query, taxonomy, concepts_terms)
            
            # Step 4: Assess domain-specific risks
            risk_assessment = await self._assess_domain_risks(query, taxonomy, methodology_map)
            
            # Step 5: Identify stakeholders and success metrics
            stakeholders_metrics = await self._identify_stakeholders_and_metrics(query, taxonomy, concepts_terms)
            
            # Step 6: Generate business context
            business_context = await self._generate_business_context(query, taxonomy, stakeholders_metrics)
            
            # Step 7: Calculate overall extraction confidence
            extraction_confidence = self._calculate_extraction_confidence(
                taxonomy, concepts_terms, methodology_map, risk_assessment
            )
            
            return EnhancedDomainKnowledge(
                taxonomy=taxonomy,
                key_concepts=concepts_terms['concepts'],
                technical_terms=concepts_terms['terms'],
                methodology_map=methodology_map,
                risk_assessment=risk_assessment,
                success_metrics=stakeholders_metrics['metrics'],
                stakeholder_map=stakeholders_metrics['stakeholders'],
                business_context=business_context,
                extraction_confidence=extraction_confidence
            )
            
        except Exception as e:
            logger.error(f"❌ Domain knowledge extraction failed: {str(e)}")
            return self._create_fallback_domain_knowledge(query)
    
    async def _classify_domain_taxonomy(self, query: str, intent_analysis: Optional[Dict] = None, data_context: Optional[Dict] = None) -> DomainTaxonomy:
        """Classify the domain taxonomy of the query"""
        
        prompt = f"""
        You are a domain classification expert. Analyze the following query and classify its domain taxonomy.
        
        Query: "{query}"
        Intent Analysis: {json.dumps(intent_analysis, indent=2) if intent_analysis else "None"}
        Data Context: {json.dumps(data_context, indent=2) if data_context else "None"}
        
        Classify the domain taxonomy:
        1. Primary domain (manufacturing, healthcare, finance, etc.)
        2. Sub-domains (specific areas within the primary domain)
        3. Industry sector (specific industry or vertical)
        4. Business function (operations, quality, engineering, etc.)
        5. Technical area (specific technical discipline)
        6. Confidence score (0.0-1.0)
        
        IMPORTANT: Respond ONLY with valid JSON in this exact format:
        {{
            "primary_domain": "manufacturing",
            "sub_domains": ["semiconductor_manufacturing", "process_control"],
            "industry_sector": "semiconductor_industry",
            "business_function": "process_engineering",
            "technical_area": "statistical_process_control",
            "confidence_score": 0.85
        }}
        """
        
        try:
            response = await asyncio.to_thread(self.llm.invoke, prompt)
            response_text = self._clean_json_response(response)
            data = json.loads(response_text)
            
            return DomainTaxonomy(
                primary_domain=DomainType(data.get("primary_domain", "general")),
                sub_domains=data.get("sub_domains", []),
                industry_sector=data.get("industry_sector", ""),
                business_function=data.get("business_function", ""),
                technical_area=data.get("technical_area", ""),
                confidence_score=float(data.get("confidence_score", 0.5))
            )
            
        except Exception as e:
            logger.error(f"Domain taxonomy classification failed: {str(e)}")
            return DomainTaxonomy(
                primary_domain=DomainType.GENERAL,
                sub_domains=["general_analysis"],
                industry_sector="general",
                business_function="analysis",
                technical_area="data_science",
                confidence_score=0.3
            )
    
    async def _extract_concepts_and_terms(self, query: str, taxonomy: DomainTaxonomy) -> Dict[str, Dict[str, KnowledgeItem]]:
        """Extract key concepts and technical terms with confidence scoring"""
        
        prompt = f"""
        You are a domain knowledge expert specializing in {taxonomy.primary_domain.value}. 
        Extract key concepts and technical terms from this query.
        
        Query: "{query}"
        Domain: {taxonomy.primary_domain.value}
        Sub-domains: {', '.join(taxonomy.sub_domains)}
        Technical Area: {taxonomy.technical_area}
        
        Extract and categorize:
        1. Key domain concepts (fundamental ideas and principles)
        2. Technical terms (specific terminology and jargon)
        
        For each item, provide:
        - The concept/term
        - Confidence level (low/medium/high/very_high)
        - Source (explicit_mention/contextual_inference/domain_taxonomy/technical_patterns)
        - Brief explanation
        - Related items
        
        IMPORTANT: Respond ONLY with valid JSON in this exact format:
        {{
            "key_concepts": [
                {{
                    "item": "process control",
                    "confidence": "high",
                    "source": "explicit_mention",
                    "explanation": "Systematic approach to managing manufacturing processes",
                    "related_items": ["quality control", "statistical monitoring"]
                }}
            ],
            "technical_terms": [
                {{
                    "item": "LOT tracking",
                    "confidence": "very_high",
                    "source": "explicit_mention",
                    "explanation": "Method for tracking production batches",
                    "related_items": ["batch processing", "traceability"]
                }}
            ]
        }}
        """
        
        try:
            response = await asyncio.to_thread(self.llm.invoke, prompt)
            response_text = self._clean_json_response(response)
            data = json.loads(response_text)
            
            concepts = {}
            terms = {}
            
            for concept_data in data.get("key_concepts", []):
                item = KnowledgeItem(
                    item=concept_data.get("item", ""),
                    confidence=KnowledgeConfidence(concept_data.get("confidence", "medium")),
                    source=KnowledgeSource(concept_data.get("source", "contextual_inference")),
                    explanation=concept_data.get("explanation", ""),
                    related_items=concept_data.get("related_items", [])
                )
                concepts[item.item] = item
            
            for term_data in data.get("technical_terms", []):
                item = KnowledgeItem(
                    item=term_data.get("item", ""),
                    confidence=KnowledgeConfidence(term_data.get("confidence", "medium")),
                    source=KnowledgeSource(term_data.get("source", "contextual_inference")),
                    explanation=term_data.get("explanation", ""),
                    related_items=term_data.get("related_items", [])
                )
                terms[item.item] = item
            
            return {"concepts": concepts, "terms": terms}
            
        except Exception as e:
            logger.error(f"Concepts and terms extraction failed: {str(e)}")
            return {
                "concepts": {"data analysis": KnowledgeItem("data analysis", KnowledgeConfidence.MEDIUM, KnowledgeSource.CONTEXTUAL_INFERENCE, "General data analysis")},
                "terms": {"statistics": KnowledgeItem("statistics", KnowledgeConfidence.MEDIUM, KnowledgeSource.CONTEXTUAL_INFERENCE, "Statistical methods")}
            }
    
    async def _map_domain_methodologies(self, query: str, taxonomy: DomainTaxonomy, concepts_terms: Dict) -> MethodologyMap:
        """Map domain-specific methodologies and best practices"""
        
        prompt = f"""
        You are a methodology expert for {taxonomy.primary_domain.value} domain.
        Map the relevant methodologies and best practices for this query.
        
        Query: "{query}"
        Domain: {taxonomy.primary_domain.value}
        Key Concepts: {', '.join(concepts_terms['concepts'].keys())}
        Technical Terms: {', '.join(concepts_terms['terms'].keys())}
        
        Identify:
        1. Standard methodologies (established approaches)
        2. Best practices (proven techniques)
        3. Tools and technologies (software, hardware, frameworks)
        4. Quality standards (industry standards, certifications)
        5. Compliance requirements (regulations, guidelines)
        
        IMPORTANT: Respond ONLY with valid JSON in this exact format:
        {{
            "standard_methodologies": ["Six Sigma", "Statistical Process Control"],
            "best_practices": ["Regular monitoring", "Root cause analysis"],
            "tools_and_technologies": ["Control charts", "SPC software"],
            "quality_standards": ["ISO 9001", "SEMI standards"],
            "compliance_requirements": ["FDA regulations", "Industry guidelines"]
        }}
        """
        
        try:
            response = await asyncio.to_thread(self.llm.invoke, prompt)
            response_text = self._clean_json_response(response)
            data = json.loads(response_text)
            
            return MethodologyMap(
                standard_methodologies=data.get("standard_methodologies", []),
                best_practices=data.get("best_practices", []),
                tools_and_technologies=data.get("tools_and_technologies", []),
                quality_standards=data.get("quality_standards", []),
                compliance_requirements=data.get("compliance_requirements", [])
            )
            
        except Exception as e:
            logger.error(f"Methodology mapping failed: {str(e)}")
            return MethodologyMap(
                standard_methodologies=["Standard analysis"],
                best_practices=["Data validation"],
                tools_and_technologies=["Statistical software"],
                quality_standards=["Industry standards"],
                compliance_requirements=["Data privacy"]
            )
    
    async def _assess_domain_risks(self, query: str, taxonomy: DomainTaxonomy, methodology_map: MethodologyMap) -> RiskAssessment:
        """Assess domain-specific risks and mitigation strategies"""
        
        prompt = f"""
        You are a risk assessment expert for {taxonomy.primary_domain.value} domain.
        Assess the potential risks for this query and suggest mitigation strategies.
        
        Query: "{query}"
        Domain: {taxonomy.primary_domain.value}
        Methodologies: {', '.join(methodology_map.standard_methodologies)}
        
        Identify risks in these categories:
        1. Technical risks (implementation, system, data risks)
        2. Business risks (cost, timeline, market risks)
        3. Operational risks (process, resource, quality risks)
        4. Compliance risks (regulatory, legal, standard risks)
        5. Mitigation strategies (preventive measures)
        
        IMPORTANT: Respond ONLY with valid JSON in this exact format:
        {{
            "technical_risks": ["Data quality issues", "System integration challenges"],
            "business_risks": ["Cost overruns", "Timeline delays"],
            "operational_risks": ["Resource constraints", "Process disruptions"],
            "compliance_risks": ["Regulatory violations", "Standard deviations"],
            "mitigation_strategies": ["Regular monitoring", "Risk assessment protocols"]
        }}
        """
        
        try:
            response = await asyncio.to_thread(self.llm.invoke, prompt)
            response_text = self._clean_json_response(response)
            data = json.loads(response_text)
            
            return RiskAssessment(
                technical_risks=data.get("technical_risks", []),
                business_risks=data.get("business_risks", []),
                operational_risks=data.get("operational_risks", []),
                compliance_risks=data.get("compliance_risks", []),
                mitigation_strategies=data.get("mitigation_strategies", [])
            )
            
        except Exception as e:
            logger.error(f"Risk assessment failed: {str(e)}")
            return RiskAssessment(
                technical_risks=["Data quality"],
                business_risks=["Resource constraints"],
                operational_risks=["Process complexity"],
                compliance_risks=["Standard compliance"],
                mitigation_strategies=["Regular monitoring"]
            )
    
    async def _identify_stakeholders_and_metrics(self, query: str, taxonomy: DomainTaxonomy, concepts_terms: Dict) -> Dict[str, any]:
        """Identify stakeholders and success metrics"""
        
        prompt = f"""
        You are a stakeholder analysis expert for {taxonomy.primary_domain.value} domain.
        Identify the stakeholders and success metrics for this query.
        
        Query: "{query}"
        Domain: {taxonomy.primary_domain.value}
        Business Function: {taxonomy.business_function}
        
        Identify:
        1. Stakeholder groups and their specific roles
        2. Success metrics that matter to each stakeholder group
        
        IMPORTANT: Respond ONLY with valid JSON in this exact format:
        {{
            "stakeholders": {{
                "primary": ["process_engineers", "quality_managers"],
                "secondary": ["production_operators", "maintenance_teams"],
                "executive": ["plant_managers", "operations_directors"]
            }},
            "success_metrics": [
                "Process stability improvement",
                "Defect rate reduction",
                "Response time to anomalies"
            ]
        }}
        """
        
        try:
            response = await asyncio.to_thread(self.llm.invoke, prompt)
            response_text = self._clean_json_response(response)
            data = json.loads(response_text)
            
            return {
                "stakeholders": data.get("stakeholders", {}),
                "metrics": data.get("success_metrics", [])
            }
            
        except Exception as e:
            logger.error(f"Stakeholder identification failed: {str(e)}")
            return {
                "stakeholders": {"primary": ["analysts"], "secondary": ["users"]},
                "metrics": ["Accuracy", "Completeness"]
            }
    
    async def _generate_business_context(self, query: str, taxonomy: DomainTaxonomy, stakeholders_metrics: Dict) -> str:
        """Generate comprehensive business context"""
        
        prompt = f"""
        You are a business context expert. Generate a comprehensive business context for this query.
        
        Query: "{query}"
        Domain: {taxonomy.primary_domain.value}
        Industry: {taxonomy.industry_sector}
        Stakeholders: {stakeholders_metrics.get('stakeholders', {})}
        
        Provide a comprehensive business context that explains:
        1. Why this analysis is important to the business
        2. How it fits into the broader business strategy
        3. What business value it delivers
        4. Who benefits and how
        
        Keep it concise but comprehensive (2-3 sentences).
        """
        
        try:
            response = await asyncio.to_thread(self.llm.invoke, prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            return response_text.strip()
            
        except Exception as e:
            logger.error(f"Business context generation failed: {str(e)}")
            return "This analysis supports data-driven decision making to improve operational efficiency and business outcomes."
    
    def _calculate_extraction_confidence(self, taxonomy: DomainTaxonomy, concepts_terms: Dict, methodology_map: MethodologyMap, risk_assessment: RiskAssessment) -> float:
        """Calculate overall extraction confidence score"""
        
        # Base confidence from taxonomy
        base_confidence = taxonomy.confidence_score
        
        # Concept confidence
        concept_confidences = [item.confidence.name for item in concepts_terms.get('concepts', {}).values()]
        term_confidences = [item.confidence.name for item in concepts_terms.get('terms', {}).values()]
        
        confidence_mapping = {"LOW": 0.25, "MEDIUM": 0.5, "HIGH": 0.75, "VERY_HIGH": 0.9}
        
        avg_concept_confidence = sum(confidence_mapping.get(c, 0.5) for c in concept_confidences) / max(len(concept_confidences), 1)
        avg_term_confidence = sum(confidence_mapping.get(c, 0.5) for c in term_confidences) / max(len(term_confidences), 1)
        
        # Methodology completeness
        methodology_completeness = min(
            len(methodology_map.standard_methodologies) / 3,
            len(methodology_map.best_practices) / 3,
            len(methodology_map.tools_and_technologies) / 3
        )
        
        # Risk assessment completeness
        risk_completeness = min(
            len(risk_assessment.technical_risks) / 2,
            len(risk_assessment.business_risks) / 2,
            len(risk_assessment.mitigation_strategies) / 2
        )
        
        # Weighted average
        overall_confidence = (
            base_confidence * 0.3 +
            avg_concept_confidence * 0.25 +
            avg_term_confidence * 0.25 +
            methodology_completeness * 0.1 +
            risk_completeness * 0.1
        )
        
        return min(overall_confidence, 1.0)
    
    def _clean_json_response(self, response) -> str:
        """Clean and prepare JSON response text"""
        response_text = response.content if hasattr(response, 'content') else str(response)
        response_text = response_text.strip()
        
        if response_text.startswith('```json'):
            response_text = response_text[7:]
        if response_text.endswith('```'):
            response_text = response_text[:-3]
        
        return response_text.strip()
    
    def _initialize_domain_patterns(self) -> Dict:
        """Initialize domain-specific patterns (can be expanded)"""
        return {
            "manufacturing": ["LOT", "process", "quality", "defect", "yield"],
            "healthcare": ["patient", "diagnosis", "treatment", "clinical"],
            "finance": ["risk", "portfolio", "return", "credit", "trading"]
        }
    
    def _initialize_methodology_database(self) -> Dict:
        """Initialize methodology database (can be expanded)"""
        return {
            "manufacturing": ["Six Sigma", "Lean", "SPC", "FMEA"],
            "healthcare": ["Clinical trials", "Evidence-based medicine"],
            "finance": ["Monte Carlo", "Black-Scholes", "VaR"]
        }
    
    def _create_fallback_domain_knowledge(self, query: str) -> EnhancedDomainKnowledge:
        """Create fallback domain knowledge when extraction fails"""
        
        return EnhancedDomainKnowledge(
            taxonomy=DomainTaxonomy(
                primary_domain=DomainType.GENERAL,
                sub_domains=["general_analysis"],
                industry_sector="general",
                business_function="analysis",
                technical_area="data_science",
                confidence_score=0.3
            ),
            key_concepts={"data analysis": KnowledgeItem("data analysis", KnowledgeConfidence.MEDIUM, KnowledgeSource.CONTEXTUAL_INFERENCE, "General data analysis")},
            technical_terms={"statistics": KnowledgeItem("statistics", KnowledgeConfidence.MEDIUM, KnowledgeSource.CONTEXTUAL_INFERENCE, "Statistical methods")},
            methodology_map=MethodologyMap(
                standard_methodologies=["Standard analysis"],
                best_practices=["Data validation"],
                tools_and_technologies=["Statistical software"],
                quality_standards=["Industry standards"],
                compliance_requirements=["Data privacy"]
            ),
            risk_assessment=RiskAssessment(
                technical_risks=["Data quality"],
                business_risks=["Resource constraints"],
                operational_risks=["Process complexity"],
                compliance_risks=["Standard compliance"],
                mitigation_strategies=["Regular monitoring"]
            ),
            success_metrics=["Accuracy", "Completeness"],
            stakeholder_map={"primary": ["analysts"], "secondary": ["users"]},
            business_context="This analysis supports data-driven decision making.",
            extraction_confidence=0.3
        )
    
    def get_domain_summary(self, domain_knowledge: EnhancedDomainKnowledge) -> str:
        """Generate human-readable summary of domain knowledge"""
        
        summary = f"""
        🎯 Domain Knowledge Extraction Summary
        
        📊 Domain Taxonomy:
        • Primary Domain: {domain_knowledge.taxonomy.primary_domain.value}
        • Industry Sector: {domain_knowledge.taxonomy.industry_sector}
        • Business Function: {domain_knowledge.taxonomy.business_function}
        • Technical Area: {domain_knowledge.taxonomy.technical_area}
        • Confidence: {domain_knowledge.taxonomy.confidence_score:.2f}
        
        🔑 Key Knowledge:
        • Concepts ({len(domain_knowledge.key_concepts)}): {', '.join(list(domain_knowledge.key_concepts.keys())[:5])}
        • Technical Terms ({len(domain_knowledge.technical_terms)}): {', '.join(list(domain_knowledge.technical_terms.keys())[:5])}
        
        📋 Methodologies:
        • Standards: {', '.join(domain_knowledge.methodology_map.standard_methodologies[:3])}
        • Best Practices: {', '.join(domain_knowledge.methodology_map.best_practices[:3])}
        
        ⚠️ Risk Assessment:
        • Technical Risks: {len(domain_knowledge.risk_assessment.technical_risks)}
        • Business Risks: {len(domain_knowledge.risk_assessment.business_risks)}
        
        👥 Stakeholders: {', '.join(domain_knowledge.stakeholder_map.get('primary', [])[:3])}
        
        💼 Business Context: {domain_knowledge.business_context}
        
        ✅ Overall Extraction Confidence: {domain_knowledge.extraction_confidence:.2f}
        """
        
        return summary.strip() 