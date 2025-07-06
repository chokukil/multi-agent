"""
Answer Structure Predictor for CherryAI

This module predicts optimal answer structures based on query types, domain knowledge,
and user context to ensure comprehensive and well-organized responses.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum

from core.llm_factory import create_llm_instance
from core.utils.logging import get_logger
from .intelligent_query_processor import AnswerFormat, AnswerStructure
from .intent_analyzer import DetailedIntentAnalysis, QueryComplexity
from .domain_extractor import EnhancedDomainKnowledge

logger = get_logger(__name__)

class VisualizationType(Enum):
    """Types of visualizations for answers"""
    # Basic Charts
    LINE_CHART = "line_chart"
    BAR_CHART = "bar_chart"
    SCATTER_PLOT = "scatter_plot"
    HISTOGRAM = "histogram"
    BOX_PLOT = "box_plot"
    
    # Process-specific
    CONTROL_CHART = "control_chart"
    PARETO_CHART = "pareto_chart"
    FISHBONE_DIAGRAM = "fishbone_diagram"
    PROCESS_FLOW = "process_flow"
    
    # Advanced Analytics
    CORRELATION_MATRIX = "correlation_matrix"
    HEATMAP = "heatmap"
    DECISION_TREE = "decision_tree"
    TIME_SERIES = "time_series"
    
    # Interactive
    DASHBOARD = "dashboard"
    INTERACTIVE_PLOT = "interactive_plot"
    DRILL_DOWN_CHART = "drill_down_chart"

class ContentSection(Enum):
    """Standard content sections for structured answers"""
    # Executive Summary
    EXECUTIVE_SUMMARY = "executive_summary"
    KEY_FINDINGS = "key_findings"
    RECOMMENDATIONS = "recommendations"
    
    # Analysis Sections
    DATA_OVERVIEW = "data_overview"
    METHODOLOGY = "methodology"
    DETAILED_ANALYSIS = "detailed_analysis"
    RESULTS_INTERPRETATION = "results_interpretation"
    
    # Domain-specific
    PROCESS_ANALYSIS = "process_analysis"
    ANOMALY_DETECTION = "anomaly_detection"
    ROOT_CAUSE_ANALYSIS = "root_cause_analysis"
    CORRECTIVE_ACTIONS = "corrective_actions"
    
    # Implementation
    IMPLEMENTATION_PLAN = "implementation_plan"
    MONITORING_STRATEGY = "monitoring_strategy"
    RISK_MITIGATION = "risk_mitigation"
    SUCCESS_METRICS = "success_metrics"
    
    # Appendix
    TECHNICAL_DETAILS = "technical_details"
    DATA_SOURCES = "data_sources"
    ASSUMPTIONS = "assumptions"
    LIMITATIONS = "limitations"

class QualityCheckpoint(Enum):
    """Quality checkpoints for answer validation"""
    DATA_VALIDATION = "data_validation"
    METHODOLOGY_REVIEW = "methodology_review"
    RESULTS_VERIFICATION = "results_verification"
    DOMAIN_EXPERT_REVIEW = "domain_expert_review"
    BUSINESS_ALIGNMENT = "business_alignment"
    COMPLETENESS_CHECK = "completeness_check"
    ACCURACY_VALIDATION = "accuracy_validation"
    STAKEHOLDER_FEEDBACK = "stakeholder_feedback"

@dataclass
class SectionSpecification:
    """Detailed specification for an answer section"""
    section_type: ContentSection
    title: str
    description: str
    required_content: List[str]
    optional_content: List[str]
    visualizations: List[VisualizationType]
    priority: int  # 1-10
    estimated_length: str  # "short", "medium", "long"
    dependencies: List[ContentSection]

@dataclass
class AnswerTemplate:
    """Template for structured answers"""
    format_type: AnswerFormat
    target_audience: str
    sections: List[SectionSpecification]
    required_visualizations: List[VisualizationType]
    quality_checkpoints: List[QualityCheckpoint]
    estimated_completion_time: str
    complexity_score: float

@dataclass
class PredictedAnswerStructure:
    """Comprehensive predicted answer structure"""
    primary_template: AnswerTemplate
    alternative_templates: List[AnswerTemplate]
    customizations: Dict[str, str]
    adaptation_reasoning: str
    confidence_score: float
    validation_criteria: List[str]

class AnswerStructurePredictor:
    """
    Advanced answer structure predictor that determines optimal response formats
    based on query analysis and domain context.
    """
    
    def __init__(self):
        self.llm = create_llm_instance()
        
        # Initialize template database
        self.template_database = self._initialize_template_database()
        self.section_library = self._initialize_section_library()
        self.visualization_rules = self._initialize_visualization_rules()
        
        logger.info("🧠 AnswerStructurePredictor initialized")
    
    async def predict_optimal_structure(self, 
                                      intent_analysis: DetailedIntentAnalysis,
                                      domain_knowledge: EnhancedDomainKnowledge,
                                      user_context: Optional[Dict] = None) -> PredictedAnswerStructure:
        """
        Predict optimal answer structure based on comprehensive analysis
        
        Args:
            intent_analysis: Detailed intent analysis results
            domain_knowledge: Comprehensive domain knowledge
            user_context: Optional user context and preferences
            
        Returns:
            PredictedAnswerStructure with detailed recommendations
        """
        logger.info(f"🔮 Predicting optimal answer structure...")
        
        try:
            # Step 1: Determine primary answer format
            primary_format = await self._determine_primary_format(intent_analysis, domain_knowledge, user_context)
            
            # Step 2: Select and customize sections
            sections = await self._select_optimal_sections(intent_analysis, domain_knowledge, primary_format)
            
            # Step 3: Determine required visualizations
            visualizations = await self._determine_visualizations(intent_analysis, domain_knowledge, sections)
            
            # Step 4: Set quality checkpoints
            checkpoints = await self._determine_quality_checkpoints(intent_analysis, domain_knowledge, primary_format)
            
            # Step 5: Create primary template
            primary_template = await self._create_answer_template(
                primary_format, intent_analysis, domain_knowledge, sections, visualizations, checkpoints
            )
            
            # Step 6: Generate alternative templates
            alternative_templates = await self._generate_alternative_templates(
                intent_analysis, domain_knowledge, primary_template
            )
            
            # Step 7: Determine customizations
            customizations = await self._determine_customizations(
                intent_analysis, domain_knowledge, user_context
            )
            
            # Step 8: Generate adaptation reasoning
            reasoning = await self._generate_adaptation_reasoning(
                intent_analysis, domain_knowledge, primary_template, customizations
            )
            
            # Step 9: Calculate confidence score
            confidence = self._calculate_structure_confidence(
                intent_analysis, domain_knowledge, primary_template
            )
            
            # Step 10: Set validation criteria
            validation_criteria = self._determine_validation_criteria(
                intent_analysis, domain_knowledge, primary_template
            )
            
            return PredictedAnswerStructure(
                primary_template=primary_template,
                alternative_templates=alternative_templates,
                customizations=customizations,
                adaptation_reasoning=reasoning,
                confidence_score=confidence,
                validation_criteria=validation_criteria
            )
            
        except Exception as e:
            logger.error(f"❌ Answer structure prediction failed: {str(e)}")
            return self._create_fallback_structure(intent_analysis, domain_knowledge)
    
    async def _determine_primary_format(self, intent_analysis: DetailedIntentAnalysis, 
                                      domain_knowledge: EnhancedDomainKnowledge, 
                                      user_context: Optional[Dict] = None) -> AnswerFormat:
        """Determine the primary answer format"""
        
        prompt = f"""
        You are an answer format expert. Determine the optimal answer format for this analysis.
        
        Query Type: {intent_analysis.query_type.value}
        Complexity: {intent_analysis.complexity_level.value}
        Urgency: {intent_analysis.urgency_level.value}
        Domain: {domain_knowledge.taxonomy.primary_domain.value}
        Business Function: {domain_knowledge.taxonomy.business_function}
        
        Primary Intent: {intent_analysis.primary_intent}
        Secondary Intents: {', '.join(intent_analysis.secondary_intents)}
        
        Available format options:
        - structured_report: Comprehensive formal report
        - visual_analysis: Visual-heavy analysis with charts
        - technical_solution: Technical implementation guide
        - business_insight: Business-focused summary
        - comparative_study: Comparison-based analysis
        - predictive_model: Predictive modeling results
        - process_optimization: Process improvement recommendations
        
        Consider:
        1. What format best serves the primary intent?
        2. What does the target audience expect?
        3. What format supports the complexity level?
        4. What format fits the domain requirements?
        
        IMPORTANT: Respond ONLY with valid JSON:
        {{
            "primary_format": "structured_report",
            "reasoning": "Explanation for format choice",
            "confidence": 0.85
        }}
        """
        
        try:
            response = await asyncio.to_thread(self.llm.invoke, prompt)
            response_text = self._clean_json_response(response)
            data = json.loads(response_text)
            
            return AnswerFormat(data.get("primary_format", "structured_report"))
            
        except Exception as e:
            logger.error(f"Primary format determination failed: {str(e)}")
            return AnswerFormat.STRUCTURED_REPORT
    
    async def _select_optimal_sections(self, intent_analysis: DetailedIntentAnalysis,
                                     domain_knowledge: EnhancedDomainKnowledge,
                                     answer_format: AnswerFormat) -> List[SectionSpecification]:
        """Select optimal sections for the answer"""
        
        prompt = f"""
        You are a content structure expert. Select the optimal sections for this answer.
        
        Query Type: {intent_analysis.query_type.value}
        Answer Format: {answer_format.value}
        Domain: {domain_knowledge.taxonomy.primary_domain.value}
        Primary Intent: {intent_analysis.primary_intent}
        
        Available section types:
        Executive: executive_summary, key_findings, recommendations
        Analysis: data_overview, methodology, detailed_analysis, results_interpretation
        Domain: process_analysis, anomaly_detection, root_cause_analysis, corrective_actions
        Implementation: implementation_plan, monitoring_strategy, risk_mitigation, success_metrics
        Appendix: technical_details, data_sources, assumptions, limitations
        
        Select 5-8 sections that best serve the intent and format.
        For each section, provide:
        - Section type
        - Custom title
        - Description
        - Priority (1-10)
        - Estimated length (short/medium/long)
        
        IMPORTANT: Respond ONLY with valid JSON:
        {{
            "sections": [
                {{
                    "section_type": "executive_summary",
                    "title": "Executive Summary",
                    "description": "High-level overview of findings",
                    "priority": 10,
                    "estimated_length": "short"
                }}
            ]
        }}
        """
        
        try:
            response = await asyncio.to_thread(self.llm.invoke, prompt)
            response_text = self._clean_json_response(response)
            data = json.loads(response_text)
            
            sections = []
            for section_data in data.get("sections", []):
                section = SectionSpecification(
                    section_type=ContentSection(section_data.get("section_type")),
                    title=section_data.get("title", ""),
                    description=section_data.get("description", ""),
                    required_content=[],
                    optional_content=[],
                    visualizations=[],
                    priority=int(section_data.get("priority", 5)),
                    estimated_length=section_data.get("estimated_length", "medium"),
                    dependencies=[]
                )
                sections.append(section)
            
            return sections
            
        except Exception as e:
            logger.error(f"Section selection failed: {str(e)}")
            return self._create_default_sections()
    
    async def _determine_visualizations(self, intent_analysis: DetailedIntentAnalysis,
                                      domain_knowledge: EnhancedDomainKnowledge,
                                      sections: List[SectionSpecification]) -> List[VisualizationType]:
        """Determine required visualizations"""
        
        prompt = f"""
        You are a data visualization expert. Determine the optimal visualizations for this analysis.
        
        Query Type: {intent_analysis.query_type.value}
        Domain: {domain_knowledge.taxonomy.primary_domain.value}
        Technical Area: {domain_knowledge.taxonomy.technical_area}
        
        Sections: {', '.join([s.section_type.value for s in sections])}
        
        Available visualization types:
        Basic: line_chart, bar_chart, scatter_plot, histogram, box_plot
        Process: control_chart, pareto_chart, fishbone_diagram, process_flow
        Advanced: correlation_matrix, heatmap, decision_tree, time_series
        Interactive: dashboard, interactive_plot, drill_down_chart
        
        Consider:
        1. What visualizations best support the analysis type?
        2. What does the domain typically use?
        3. What helps explain complex concepts?
        4. What supports decision making?
        
        Select 3-7 visualization types that best serve the analysis.
        
        IMPORTANT: Respond ONLY with valid JSON:
        {{
            "visualizations": ["control_chart", "pareto_chart", "line_chart"],
            "reasoning": "Explanation for visualization choices"
        }}
        """
        
        try:
            response = await asyncio.to_thread(self.llm.invoke, prompt)
            response_text = self._clean_json_response(response)
            data = json.loads(response_text)
            
            visualizations = []
            for viz_name in data.get("visualizations", []):
                try:
                    visualizations.append(VisualizationType(viz_name))
                except ValueError:
                    logger.warning(f"Unknown visualization type: {viz_name}")
            
            return visualizations
            
        except Exception as e:
            logger.error(f"Visualization determination failed: {str(e)}")
            return [VisualizationType.LINE_CHART, VisualizationType.BAR_CHART]
    
    async def _determine_quality_checkpoints(self, intent_analysis: DetailedIntentAnalysis,
                                           domain_knowledge: EnhancedDomainKnowledge,
                                           answer_format: AnswerFormat) -> List[QualityCheckpoint]:
        """Determine quality checkpoints"""
        
        # Base checkpoints for all analyses
        checkpoints = [
            QualityCheckpoint.DATA_VALIDATION,
            QualityCheckpoint.METHODOLOGY_REVIEW,
            QualityCheckpoint.RESULTS_VERIFICATION
        ]
        
        # Add domain-specific checkpoints
        if domain_knowledge.taxonomy.primary_domain.value in ["manufacturing", "healthcare"]:
            checkpoints.append(QualityCheckpoint.DOMAIN_EXPERT_REVIEW)
        
        # Add complexity-based checkpoints
        if intent_analysis.complexity_level in [QueryComplexity.COMPLEX, QueryComplexity.VERY_COMPLEX]:
            checkpoints.extend([
                QualityCheckpoint.BUSINESS_ALIGNMENT,
                QualityCheckpoint.COMPLETENESS_CHECK
            ])
        
        # Add format-specific checkpoints
        if answer_format in [AnswerFormat.BUSINESS_INSIGHT, AnswerFormat.PROCESS_OPTIMIZATION]:
            checkpoints.append(QualityCheckpoint.STAKEHOLDER_FEEDBACK)
        
        return list(set(checkpoints))  # Remove duplicates
    
    async def _create_answer_template(self, format_type: AnswerFormat,
                                    intent_analysis: DetailedIntentAnalysis,
                                    domain_knowledge: EnhancedDomainKnowledge,
                                    sections: List[SectionSpecification],
                                    visualizations: List[VisualizationType],
                                    checkpoints: List[QualityCheckpoint]) -> AnswerTemplate:
        """Create comprehensive answer template"""
        
        # Determine target audience
        stakeholders = domain_knowledge.stakeholder_map
        if "executive" in stakeholders and stakeholders["executive"]:
            target_audience = "executive"
        elif "primary" in stakeholders and stakeholders["primary"]:
            target_audience = "technical"
        else:
            target_audience = "general"
        
        # Estimate completion time based on complexity
        complexity_time_map = {
            QueryComplexity.SIMPLE: "2-4 hours",
            QueryComplexity.MODERATE: "4-8 hours",
            QueryComplexity.COMPLEX: "1-2 days",
            QueryComplexity.VERY_COMPLEX: "2-5 days"
        }
        completion_time = complexity_time_map.get(intent_analysis.complexity_level, "1 day")
        
        # Calculate complexity score
        complexity_score = self._calculate_complexity_score(intent_analysis, domain_knowledge, sections)
        
        return AnswerTemplate(
            format_type=format_type,
            target_audience=target_audience,
            sections=sections,
            required_visualizations=visualizations,
            quality_checkpoints=checkpoints,
            estimated_completion_time=completion_time,
            complexity_score=complexity_score
        )
    
    async def _generate_alternative_templates(self, intent_analysis: DetailedIntentAnalysis,
                                            domain_knowledge: EnhancedDomainKnowledge,
                                            primary_template: AnswerTemplate) -> List[AnswerTemplate]:
        """Generate alternative answer templates"""
        
        alternatives = []
        
        # Create simplified version for quick delivery
        if primary_template.complexity_score > 0.7:
            simplified_sections = [s for s in primary_template.sections if s.priority >= 7]
            simplified_template = AnswerTemplate(
                format_type=AnswerFormat.BUSINESS_INSIGHT,
                target_audience="executive",
                sections=simplified_sections,
                required_visualizations=primary_template.required_visualizations[:3],
                quality_checkpoints=[QualityCheckpoint.BUSINESS_ALIGNMENT],
                estimated_completion_time="2-4 hours",
                complexity_score=0.4
            )
            alternatives.append(simplified_template)
        
        # Create detailed version for comprehensive analysis
        if primary_template.format_type != AnswerFormat.TECHNICAL_SOLUTION:
            detailed_template = AnswerTemplate(
                format_type=AnswerFormat.TECHNICAL_SOLUTION,
                target_audience="technical",
                sections=primary_template.sections + self._get_technical_sections(),
                required_visualizations=primary_template.required_visualizations,
                quality_checkpoints=primary_template.quality_checkpoints + [QualityCheckpoint.DOMAIN_EXPERT_REVIEW],
                estimated_completion_time="2-3 days",
                complexity_score=min(primary_template.complexity_score + 0.2, 1.0)
            )
            alternatives.append(detailed_template)
        
        return alternatives
    
    async def _determine_customizations(self, intent_analysis: DetailedIntentAnalysis,
                                      domain_knowledge: EnhancedDomainKnowledge,
                                      user_context: Optional[Dict] = None) -> Dict[str, str]:
        """Determine answer customizations"""
        
        customizations = {}
        
        # Domain-specific customizations
        if domain_knowledge.taxonomy.primary_domain.value == "manufacturing":
            customizations["terminology"] = "Use manufacturing-specific terms and metrics"
            customizations["focus"] = "Emphasize process efficiency and quality"
        
        # Urgency-based customizations
        if intent_analysis.urgency_level.value in ["high", "critical"]:
            customizations["delivery"] = "Prioritize quick wins and immediate actions"
            customizations["format"] = "Include executive summary at the beginning"
        
        # Complexity-based customizations
        if intent_analysis.complexity_level == QueryComplexity.VERY_COMPLEX:
            customizations["structure"] = "Break into phases with interim deliverables"
            customizations["validation"] = "Include multiple review checkpoints"
        
        return customizations
    
    async def _generate_adaptation_reasoning(self, intent_analysis: DetailedIntentAnalysis,
                                           domain_knowledge: EnhancedDomainKnowledge,
                                           template: AnswerTemplate,
                                           customizations: Dict[str, str]) -> str:
        """Generate reasoning for structure adaptation"""
        
        reasoning_parts = [
            f"Selected {template.format_type.value} format based on {intent_analysis.query_type.value} query type",
            f"Targeted {template.target_audience} audience given stakeholder priorities",
            f"Included {len(template.sections)} sections to address {intent_analysis.complexity_level.value} complexity",
            f"Added {len(template.required_visualizations)} visualizations for {domain_knowledge.taxonomy.technical_area}",
        ]
        
        if customizations:
            reasoning_parts.append(f"Applied customizations: {', '.join(customizations.keys())}")
        
        return ". ".join(reasoning_parts) + "."
    
    def _calculate_structure_confidence(self, intent_analysis: DetailedIntentAnalysis,
                                      domain_knowledge: EnhancedDomainKnowledge,
                                      template: AnswerTemplate) -> float:
        """Calculate confidence in structure prediction"""
        
        base_confidence = intent_analysis.overall_confidence
        domain_confidence = domain_knowledge.extraction_confidence
        
        # Adjust based on template complexity alignment
        complexity_alignment = 1.0 - abs(template.complexity_score - 
                                       self._get_intent_complexity_score(intent_analysis))
        
        # Adjust based on section completeness
        section_completeness = min(len(template.sections) / 6, 1.0)  # Target 6 sections
        
        overall_confidence = (
            base_confidence * 0.4 +
            domain_confidence * 0.3 +
            complexity_alignment * 0.2 +
            section_completeness * 0.1
        )
        
        return min(overall_confidence, 1.0)
    
    def _determine_validation_criteria(self, intent_analysis: DetailedIntentAnalysis,
                                     domain_knowledge: EnhancedDomainKnowledge,
                                     template: AnswerTemplate) -> List[str]:
        """Determine validation criteria for the answer"""
        
        criteria = [
            "All required sections are complete and comprehensive",
            "Visualizations clearly support the analysis findings",
            "Recommendations are actionable and specific",
            "Technical accuracy is verified by domain experts"
        ]
        
        # Add domain-specific criteria
        if domain_knowledge.taxonomy.primary_domain.value == "manufacturing":
            criteria.append("Process improvements are feasible and cost-effective")
            criteria.append("Quality metrics show measurable improvement potential")
        
        # Add complexity-specific criteria
        if intent_analysis.complexity_level in [QueryComplexity.COMPLEX, QueryComplexity.VERY_COMPLEX]:
            criteria.append("Complex concepts are explained clearly for target audience")
            criteria.append("Implementation plan includes risk mitigation strategies")
        
        return criteria
    
    def _clean_json_response(self, response) -> str:
        """Clean and prepare JSON response text"""
        response_text = response.content if hasattr(response, 'content') else str(response)
        response_text = response_text.strip()
        
        if response_text.startswith('```json'):
            response_text = response_text[7:]
        if response_text.endswith('```'):
            response_text = response_text[:-3]
        
        return response_text.strip()
    
    def _initialize_template_database(self) -> Dict:
        """Initialize template database"""
        return {
            "manufacturing": {
                "diagnostic": ["process_analysis", "root_cause_analysis", "corrective_actions"],
                "analytical": ["data_overview", "detailed_analysis", "results_interpretation"]
            }
        }
    
    def _initialize_section_library(self) -> Dict:
        """Initialize section library"""
        return {
            ContentSection.EXECUTIVE_SUMMARY: "High-level overview and key findings",
            ContentSection.PROCESS_ANALYSIS: "Detailed process performance analysis",
            ContentSection.ROOT_CAUSE_ANALYSIS: "Investigation of underlying causes"
        }
    
    def _initialize_visualization_rules(self) -> Dict:
        """Initialize visualization rules"""
        return {
            "manufacturing": [VisualizationType.CONTROL_CHART, VisualizationType.PARETO_CHART],
            "time_series": [VisualizationType.LINE_CHART, VisualizationType.TIME_SERIES],
            "comparison": [VisualizationType.BAR_CHART, VisualizationType.BOX_PLOT]
        }
    
    def _create_default_sections(self) -> List[SectionSpecification]:
        """Create default sections when selection fails"""
        return [
            SectionSpecification(
                section_type=ContentSection.EXECUTIVE_SUMMARY,
                title="Executive Summary",
                description="High-level overview of findings",
                required_content=[],
                optional_content=[],
                visualizations=[],
                priority=10,
                estimated_length="short",
                dependencies=[]
            ),
            SectionSpecification(
                section_type=ContentSection.DETAILED_ANALYSIS,
                title="Analysis Results",
                description="Detailed analysis and findings",
                required_content=[],
                optional_content=[],
                visualizations=[],
                priority=8,
                estimated_length="long",
                dependencies=[]
            )
        ]
    
    def _get_technical_sections(self) -> List[SectionSpecification]:
        """Get additional technical sections"""
        return [
            SectionSpecification(
                section_type=ContentSection.TECHNICAL_DETAILS,
                title="Technical Implementation Details",
                description="Detailed technical specifications",
                required_content=[],
                optional_content=[],
                visualizations=[],
                priority=6,
                estimated_length="medium",
                dependencies=[]
            )
        ]
    
    def _calculate_complexity_score(self, intent_analysis: DetailedIntentAnalysis,
                                  domain_knowledge: EnhancedDomainKnowledge,
                                  sections: List[SectionSpecification]) -> float:
        """Calculate template complexity score"""
        
        intent_complexity = self._get_intent_complexity_score(intent_analysis)
        domain_complexity = 1.0 - domain_knowledge.extraction_confidence
        section_complexity = len(sections) / 10  # Normalize by max expected sections
        
        return (intent_complexity * 0.5 + domain_complexity * 0.3 + section_complexity * 0.2)
    
    def _get_intent_complexity_score(self, intent_analysis: DetailedIntentAnalysis) -> float:
        """Get complexity score from intent analysis"""
        complexity_map = {
            QueryComplexity.SIMPLE: 0.2,
            QueryComplexity.MODERATE: 0.5,
            QueryComplexity.COMPLEX: 0.8,
            QueryComplexity.VERY_COMPLEX: 1.0
        }
        return complexity_map.get(intent_analysis.complexity_level, 0.5)
    
    def _create_fallback_structure(self, intent_analysis: DetailedIntentAnalysis,
                                 domain_knowledge: EnhancedDomainKnowledge) -> PredictedAnswerStructure:
        """Create fallback structure when prediction fails"""
        
        fallback_sections = self._create_default_sections()
        fallback_template = AnswerTemplate(
            format_type=AnswerFormat.STRUCTURED_REPORT,
            target_audience="general",
            sections=fallback_sections,
            required_visualizations=[VisualizationType.BAR_CHART],
            quality_checkpoints=[QualityCheckpoint.DATA_VALIDATION],
            estimated_completion_time="1 day",
            complexity_score=0.5
        )
        
        return PredictedAnswerStructure(
            primary_template=fallback_template,
            alternative_templates=[],
            customizations={},
            adaptation_reasoning="Fallback structure due to prediction failure",
            confidence_score=0.3,
            validation_criteria=["Basic completeness check"]
        )
    
    def get_structure_summary(self, predicted_structure: PredictedAnswerStructure) -> str:
        """Generate human-readable summary of predicted structure"""
        
        template = predicted_structure.primary_template
        
        summary = f"""
        🎯 Predicted Answer Structure Summary
        
        📋 Primary Template:
        • Format: {template.format_type.value}
        • Target Audience: {template.target_audience}
        • Sections ({len(template.sections)}): {', '.join([s.title for s in template.sections[:5]])}
        • Visualizations ({len(template.required_visualizations)}): {', '.join([v.value for v in template.required_visualizations[:5]])}
        
        ⏱️ Execution Details:
        • Estimated Time: {template.estimated_completion_time}
        • Complexity Score: {template.complexity_score:.2f}
        • Quality Checkpoints: {len(template.quality_checkpoints)}
        
        🔄 Alternatives: {len(predicted_structure.alternative_templates)} templates available
        
        🎛️ Customizations: {', '.join(predicted_structure.customizations.keys()) if predicted_structure.customizations else 'None'}
        
        💡 Adaptation Reasoning: {predicted_structure.adaptation_reasoning}
        
        ✅ Prediction Confidence: {predicted_structure.confidence_score:.2f}
        """
        
        return summary.strip() 