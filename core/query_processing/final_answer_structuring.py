"""
Final Answer Structuring Module

This module provides the final integration point for all Phase 3 components,
creating comprehensive, structured answers that combine holistic synthesis,
domain-specific formatting, personalized optimization, and quality validation
into a single, coherent response.

Author: CherryAI Development Team
Version: 1.0.0
"""

import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json
import hashlib
import uuid

# Phase 3 imports
from .holistic_answer_synthesis_engine import HolisticAnswer, AnswerSection, AnswerStyle
from .domain_specific_answer_formatter import FormattedAnswer, OutputFormat, DomainType, FormattingStyle
from .user_personalized_result_optimizer import OptimizedResult, PersonalizationLevel, UserRole
from .answer_quality_validator import QualityReport, QualityLevel, QualityMetric

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StructureType(Enum):
    """Types of final answer structures"""
    EXECUTIVE_REPORT = "executive_report"        # Executive summary format
    TECHNICAL_DOCUMENT = "technical_document"   # Technical documentation format
    PRESENTATION = "presentation"               # Presentation slide format
    DASHBOARD = "dashboard"                     # Dashboard/metrics format
    NARRATIVE = "narrative"                     # Story-like narrative format
    ACADEMIC_PAPER = "academic_paper"           # Academic research format
    CONSULTATION = "consultation"               # Consulting advice format
    COMPREHENSIVE = "comprehensive"             # All-inclusive format


class ExportFormat(Enum):
    """Export formats for final answers"""
    JSON = "json"
    MARKDOWN = "markdown"
    HTML = "html"
    PDF = "pdf"
    DOCX = "docx"
    POWERPOINT = "powerpoint"
    PLAIN_TEXT = "plain_text"
    STRUCTURED_DATA = "structured_data"


class PresentationMode(Enum):
    """Presentation modes for different contexts"""
    INTERACTIVE = "interactive"                 # Interactive web format
    STATIC = "static"                          # Static document format
    STREAMING = "streaming"                    # Real-time streaming format
    PRINT_FRIENDLY = "print_friendly"          # Print-optimized format
    MOBILE_OPTIMIZED = "mobile_optimized"      # Mobile device format


@dataclass
class AnswerMetadata:
    """Comprehensive metadata for final answers"""
    answer_id: str
    session_id: str
    user_id: str
    query_hash: str
    generation_timestamp: datetime
    processing_time: float
    component_versions: Dict[str, str] = field(default_factory=dict)
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    personalization_data: Dict[str, Any] = field(default_factory=dict)
    source_tracking: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StructuringContext:
    """Context for final answer structuring"""
    structure_type: StructureType
    export_format: ExportFormat
    presentation_mode: PresentationMode
    target_audience: str
    use_case: str
    branding_preferences: Dict[str, Any] = field(default_factory=dict)
    layout_preferences: Dict[str, Any] = field(default_factory=dict)
    content_preferences: Dict[str, Any] = field(default_factory=dict)
    accessibility_requirements: List[str] = field(default_factory=list)


@dataclass
class AnswerSection:
    """Individual section in final structured answer"""
    section_id: str
    title: str
    content: str
    section_type: str
    priority: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    subsections: List['AnswerSection'] = field(default_factory=list)


@dataclass
class AnswerComponent:
    """Individual component in final answer"""
    component_id: str
    component_type: str
    title: str
    content: Any
    styling: Dict[str, Any] = field(default_factory=dict)
    interactive_elements: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FinalStructuredAnswer:
    """Complete structured answer with all components"""
    answer_id: str
    title: str
    executive_summary: str
    main_sections: List[AnswerSection]
    components: List[AnswerComponent]
    quality_report: QualityReport
    metadata: AnswerMetadata
    structure_type: StructureType
    export_format: ExportFormat
    presentation_mode: PresentationMode
    
    # Source components
    holistic_answer: HolisticAnswer
    formatted_answer: FormattedAnswer
    optimized_result: OptimizedResult
    
    # Additional features
    interactive_elements: List[Dict[str, Any]] = field(default_factory=list)
    visualizations: List[Dict[str, Any]] = field(default_factory=list)
    appendices: List[Dict[str, Any]] = field(default_factory=list)
    citations: List[str] = field(default_factory=list)
    
    generated_at: datetime = field(default_factory=datetime.now)


class FinalAnswerStructuring:
    """
    Final answer structuring engine that combines all Phase 3 components
    into comprehensive, structured responses
    """
    
    def __init__(self):
        """Initialize the Final Answer Structuring engine"""
        self.logger = logging.getLogger(__name__)
        self.structure_templates = self._initialize_structure_templates()
        self.formatting_rules = self._initialize_formatting_rules()
        self.presentation_styles = self._initialize_presentation_styles()
        self.structuring_history: List[FinalStructuredAnswer] = []
        
    def _initialize_structure_templates(self) -> Dict[StructureType, Dict[str, Any]]:
        """Initialize structure templates for different answer types"""
        return {
            StructureType.EXECUTIVE_REPORT: {
                "sections": [
                    {"id": "executive_summary", "title": "Executive Summary", "priority": 1},
                    {"id": "key_findings", "title": "Key Findings", "priority": 2},
                    {"id": "recommendations", "title": "Recommendations", "priority": 3},
                    {"id": "next_steps", "title": "Next Steps", "priority": 4},
                    {"id": "supporting_data", "title": "Supporting Data", "priority": 5}
                ],
                "style": {
                    "concise": True,
                    "action_oriented": True,
                    "executive_language": True
                }
            },
            StructureType.TECHNICAL_DOCUMENT: {
                "sections": [
                    {"id": "overview", "title": "Technical Overview", "priority": 1},
                    {"id": "methodology", "title": "Methodology", "priority": 2},
                    {"id": "detailed_analysis", "title": "Detailed Analysis", "priority": 3},
                    {"id": "implementation", "title": "Implementation", "priority": 4},
                    {"id": "technical_specifications", "title": "Technical Specifications", "priority": 5},
                    {"id": "appendices", "title": "Technical Appendices", "priority": 6}
                ],
                "style": {
                    "detailed": True,
                    "technical_language": True,
                    "code_examples": True,
                    "diagrams": True
                }
            },
            StructureType.PRESENTATION: {
                "sections": [
                    {"id": "title_slide", "title": "Title & Agenda", "priority": 1},
                    {"id": "problem_statement", "title": "Problem Statement", "priority": 2},
                    {"id": "analysis", "title": "Analysis", "priority": 3},
                    {"id": "solutions", "title": "Solutions", "priority": 4},
                    {"id": "recommendations", "title": "Recommendations", "priority": 5},
                    {"id": "qa", "title": "Q&A", "priority": 6}
                ],
                "style": {
                    "visual": True,
                    "bullet_points": True,
                    "slide_format": True
                }
            },
            StructureType.DASHBOARD: {
                "sections": [
                    {"id": "kpi_overview", "title": "Key Performance Indicators", "priority": 1},
                    {"id": "metrics", "title": "Metrics & Analytics", "priority": 2},
                    {"id": "trends", "title": "Trends & Patterns", "priority": 3},
                    {"id": "alerts", "title": "Alerts & Notifications", "priority": 4},
                    {"id": "actions", "title": "Recommended Actions", "priority": 5}
                ],
                "style": {
                    "visual": True,
                    "metrics_focused": True,
                    "interactive": True
                }
            },
            StructureType.COMPREHENSIVE: {
                "sections": [
                    {"id": "executive_summary", "title": "Executive Summary", "priority": 1},
                    {"id": "detailed_analysis", "title": "Detailed Analysis", "priority": 2},
                    {"id": "methodology", "title": "Methodology", "priority": 3},
                    {"id": "findings", "title": "Key Findings", "priority": 4},
                    {"id": "recommendations", "title": "Recommendations", "priority": 5},
                    {"id": "implementation", "title": "Implementation Plan", "priority": 6},
                    {"id": "quality_assessment", "title": "Quality Assessment", "priority": 7},
                    {"id": "appendices", "title": "Appendices", "priority": 8}
                ],
                "style": {
                    "comprehensive": True,
                    "all_inclusive": True,
                    "detailed": True
                }
            }
        }
    
    def _initialize_formatting_rules(self) -> Dict[ExportFormat, Dict[str, Any]]:
        """Initialize formatting rules for different export formats"""
        return {
            ExportFormat.MARKDOWN: {
                "headers": {"h1": "#", "h2": "##", "h3": "###"},
                "emphasis": {"bold": "**", "italic": "*", "code": "`"},
                "lists": {"unordered": "-", "ordered": "1."},
                "links": "[text](url)",
                "images": "![alt](url)",
                "tables": "|---|---|---|",
                "code_blocks": "```language\ncode\n```"
            },
            ExportFormat.HTML: {
                "headers": {"h1": "<h1>", "h2": "<h2>", "h3": "<h3>"},
                "emphasis": {"bold": "<strong>", "italic": "<em>", "code": "<code>"},
                "lists": {"unordered": "<ul><li>", "ordered": "<ol><li>"},
                "links": "<a href='url'>text</a>",
                "images": "<img src='url' alt='alt'>",
                "tables": "<table><tr><td>",
                "code_blocks": "<pre><code class='language'>"
            },
            ExportFormat.JSON: {
                "structure": "hierarchical",
                "metadata": "included",
                "formatting": "preserved",
                "validation": "strict"
            }
        }
    
    def _initialize_presentation_styles(self) -> Dict[PresentationMode, Dict[str, Any]]:
        """Initialize presentation styles for different modes"""
        return {
            PresentationMode.INTERACTIVE: {
                "features": ["clickable_sections", "expandable_content", "tooltips", "navigation"],
                "styling": {"responsive": True, "animated": True, "interactive_elements": True}
            },
            PresentationMode.STATIC: {
                "features": ["fixed_layout", "print_ready", "clear_typography"],
                "styling": {"responsive": False, "animated": False, "static_elements": True}
            },
            PresentationMode.MOBILE_OPTIMIZED: {
                "features": ["touch_friendly", "compact_layout", "fast_loading"],
                "styling": {"responsive": True, "mobile_first": True, "touch_optimized": True}
            }
        }
    
    def structure_final_answer(
        self,
        holistic_answer: HolisticAnswer,
        formatted_answer: FormattedAnswer,
        optimized_result: OptimizedResult,
        quality_report: QualityReport,
        structuring_context: StructuringContext,
        user_id: str = "anonymous",
        session_id: str = None
    ) -> FinalStructuredAnswer:
        """
        Create final structured answer by combining all Phase 3 components
        
        Args:
            holistic_answer: Output from Phase 3.1
            formatted_answer: Output from Phase 3.2
            optimized_result: Output from Phase 3.3
            quality_report: Output from Phase 3.4
            structuring_context: Context for final structuring
            user_id: User identifier
            session_id: Session identifier
            
        Returns:
            FinalStructuredAnswer: Complete structured answer
        """
        start_time = datetime.now()
        answer_id = str(uuid.uuid4())
        
        logger.info(f"🔄 Starting final answer structuring: {answer_id}")
        
        try:
            # 1. Create answer metadata
            metadata = self._create_answer_metadata(
                answer_id, user_id, session_id, 
                holistic_answer, quality_report, start_time
            )
            
            # 2. Generate title
            title = self._generate_answer_title(
                holistic_answer, formatted_answer, structuring_context
            )
            
            # 3. Create executive summary
            executive_summary = self._create_executive_summary(
                holistic_answer, optimized_result, quality_report, structuring_context
            )
            
            # 4. Structure main sections
            main_sections = self._structure_main_sections(
                holistic_answer, formatted_answer, optimized_result, structuring_context
            )
            
            # 5. Create answer components
            components = self._create_answer_components(
                holistic_answer, formatted_answer, optimized_result, structuring_context
            )
            
            # 6. Generate interactive elements
            interactive_elements = self._generate_interactive_elements(
                holistic_answer, structuring_context
            )
            
            # 7. Create visualizations
            visualizations = self._create_visualizations(
                holistic_answer, formatted_answer, structuring_context
            )
            
            # 8. Generate appendices
            appendices = self._generate_appendices(
                holistic_answer, formatted_answer, quality_report, structuring_context
            )
            
            # 9. Compile citations
            citations = self._compile_citations(
                holistic_answer, formatted_answer, quality_report
            )
            
            # 10. Create final structured answer
            final_answer = FinalStructuredAnswer(
                answer_id=answer_id,
                title=title,
                executive_summary=executive_summary,
                main_sections=main_sections,
                components=components,
                quality_report=quality_report,
                metadata=metadata,
                structure_type=structuring_context.structure_type,
                export_format=structuring_context.export_format,
                presentation_mode=structuring_context.presentation_mode,
                holistic_answer=holistic_answer,
                formatted_answer=formatted_answer,
                optimized_result=optimized_result,
                interactive_elements=interactive_elements,
                visualizations=visualizations,
                appendices=appendices,
                citations=citations,
                generated_at=datetime.now()
            )
            
            # 11. Apply final formatting
            formatted_final_answer = self._apply_final_formatting(
                final_answer, structuring_context
            )
            
            # 12. Store in history
            self.structuring_history.append(formatted_final_answer)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"✅ Final answer structuring completed: {answer_id} ({processing_time:.2f}s)")
            
            return formatted_final_answer
            
        except Exception as e:
            logger.error(f"❌ Error in final answer structuring: {str(e)}")
            raise
    
    def _create_answer_metadata(
        self,
        answer_id: str,
        user_id: str,
        session_id: str,
        holistic_answer: HolisticAnswer,
        quality_report: QualityReport,
        start_time: datetime
    ) -> AnswerMetadata:
        """Create comprehensive metadata for the final answer"""
        
        # Create query hash for tracking
        query_hash = hashlib.md5(
            holistic_answer.query_summary.encode()
        ).hexdigest()
        
        return AnswerMetadata(
            answer_id=answer_id,
            session_id=session_id or f"session_{int(start_time.timestamp())}",
            user_id=user_id,
            query_hash=query_hash,
            generation_timestamp=start_time,
            processing_time=0.0,  # Will be updated at the end
            component_versions={
                "holistic_synthesis": "3.1.0",
                "domain_formatting": "3.2.0", 
                "personalization": "3.3.0",
                "quality_validation": "3.4.0",
                "final_structuring": "3.5.0"
            },
            quality_metrics={
                "overall_score": quality_report.overall_score,
                "quality_level": quality_report.overall_level.value,
                "passed_validation": quality_report.passed_validation
            },
            personalization_data={
                "optimization_applied": True,
                "personalization_level": "advanced"
            },
            source_tracking={
                "holistic_answer_id": holistic_answer.answer_id,
                "quality_report_timestamp": quality_report.validation_timestamp
            }
        )
    
    def _generate_answer_title(
        self,
        holistic_answer: HolisticAnswer,
        formatted_answer: FormattedAnswer,
        context: StructuringContext
    ) -> str:
        """Generate appropriate title for the final answer"""
        
        base_title = holistic_answer.query_summary
        
        # Customize title based on structure type
        if context.structure_type == StructureType.EXECUTIVE_REPORT:
            return f"Executive Analysis: {base_title}"
        elif context.structure_type == StructureType.TECHNICAL_DOCUMENT:
            return f"Technical Documentation: {base_title}"
        elif context.structure_type == StructureType.PRESENTATION:
            return f"Presentation: {base_title}"
        elif context.structure_type == StructureType.DASHBOARD:
            return f"Analytics Dashboard: {base_title}"
        elif context.structure_type == StructureType.COMPREHENSIVE:
            return f"Comprehensive Analysis: {base_title}"
        else:
            return base_title
    
    def _create_executive_summary(
        self,
        holistic_answer: HolisticAnswer,
        optimized_result: OptimizedResult,
        quality_report: QualityReport,
        context: StructuringContext
    ) -> str:
        """Create executive summary combining all insights"""
        
        # Start with holistic summary
        base_summary = holistic_answer.executive_summary
        
        # Add quality indicators
        quality_indicator = f"(Quality Score: {quality_report.overall_score:.1f}/1.0)"
        
        # Add personalization note if applied
        personalization_note = ""
        if len(optimized_result.personalization_insights.applied_optimizations) > 0:
            personalization_note = f" This analysis has been personalized based on your preferences and role."
        
        return f"{base_summary} {quality_indicator}{personalization_note}"
    
    def _structure_main_sections(
        self,
        holistic_answer: HolisticAnswer,
        formatted_answer: FormattedAnswer,
        optimized_result: OptimizedResult,
        context: StructuringContext
    ) -> List[AnswerSection]:
        """Structure main sections based on context and template"""
        
        sections = []
        template = self.structure_templates.get(context.structure_type, {})
        section_configs = template.get("sections", [])
        
        for i, section_config in enumerate(section_configs):
            section_content = self._generate_section_content(
                section_config, holistic_answer, formatted_answer, optimized_result
            )
            
            section = AnswerSection(
                section_id=section_config["id"],
                title=section_config["title"],
                content=section_content,
                section_type=section_config["id"],
                priority=section_config["priority"],
                metadata={
                    "source": "integrated",
                    "template": context.structure_type.value,
                    "order": i + 1
                }
            )
            
            sections.append(section)
        
        return sections
    
    def _generate_section_content(
        self,
        section_config: Dict[str, Any],
        holistic_answer: HolisticAnswer,
        formatted_answer: FormattedAnswer,
        optimized_result: OptimizedResult
    ) -> str:
        """Generate content for a specific section"""
        
        section_id = section_config["id"]
        
        if section_id == "executive_summary":
            return holistic_answer.executive_summary
        elif section_id == "key_findings" or section_id == "findings":
            return "\n".join([f"• {insight}" for insight in holistic_answer.key_insights])
        elif section_id == "recommendations":
            return "\n".join([f"{i+1}. {rec}" for i, rec in enumerate(holistic_answer.recommendations)])
        elif section_id == "next_steps":
            return "\n".join([f"• {step}" for step in holistic_answer.next_steps])
        elif section_id == "detailed_analysis":
            # Combine sections from holistic answer
            analysis_content = []
            for section in holistic_answer.main_sections:
                if section.section_type == "analysis":
                    analysis_content.append(f"## {section.title}\n{section.content}")
            return "\n\n".join(analysis_content)
        elif section_id == "methodology":
            # Look for methodology in sections
            for section in holistic_answer.main_sections:
                if "methodology" in section.title.lower() or "method" in section.title.lower():
                    return section.content
            return "Methodology details are integrated within the analysis sections."
        elif section_id == "supporting_data":
            return "Supporting data and evidence have been integrated throughout the analysis."
        elif section_id == "quality_assessment":
            return f"This analysis has achieved a quality score of {holistic_answer.confidence_score:.1f}/1.0 based on comprehensive validation."
        else:
            # Default to formatted content
            return formatted_answer.content[:500] + "..." if len(formatted_answer.content) > 500 else formatted_answer.content
    
    def _create_answer_components(
        self,
        holistic_answer: HolisticAnswer,
        formatted_answer: FormattedAnswer,
        optimized_result: OptimizedResult,
        context: StructuringContext
    ) -> List[AnswerComponent]:
        """Create structured components for the final answer"""
        
        components = []
        
        # Main content component
        main_component = AnswerComponent(
            component_id="main_content",
            component_type="content",
            title="Main Analysis",
            content=optimized_result.optimized_content,
            styling={"format": context.export_format.value},
            metadata={"source": "optimized_result"}
        )
        components.append(main_component)
        
        # Quality metrics component
        quality_component = AnswerComponent(
            component_id="quality_metrics",
            component_type="metrics",
            title="Quality Metrics",
            content=holistic_answer.quality_metrics,
            styling={"display": "chart", "format": "json"},
            metadata={"source": "holistic_answer"}
        )
        components.append(quality_component)
        
        # Insights component
        insights_component = AnswerComponent(
            component_id="key_insights",
            component_type="insights",
            title="Key Insights",
            content=holistic_answer.key_insights,
            styling={"display": "list", "emphasis": "highlight"},
            metadata={"source": "holistic_answer"}
        )
        components.append(insights_component)
        
        return components
    
    def _generate_interactive_elements(
        self,
        holistic_answer: HolisticAnswer,
        context: StructuringContext
    ) -> List[Dict[str, Any]]:
        """Generate interactive elements based on presentation mode"""
        
        if context.presentation_mode != PresentationMode.INTERACTIVE:
            return []
        
        elements = []
        
        # Expandable sections
        elements.append({
            "type": "expandable_section",
            "title": "Detailed Analysis",
            "content": "Click to expand detailed analysis",
            "target": "detailed_analysis_section"
        })
        
        # Interactive insights
        elements.append({
            "type": "interactive_insights",
            "title": "Key Insights Explorer",
            "content": holistic_answer.key_insights,
            "interaction": "hover_details"
        })
        
        # Action buttons
        elements.append({
            "type": "action_buttons",
            "actions": [
                {"label": "Export PDF", "action": "export_pdf"},
                {"label": "Share Analysis", "action": "share"},
                {"label": "Schedule Review", "action": "schedule"}
            ]
        })
        
        return elements
    
    def _create_visualizations(
        self,
        holistic_answer: HolisticAnswer,
        formatted_answer: FormattedAnswer,
        context: StructuringContext
    ) -> List[Dict[str, Any]]:
        """Create visualizations for the final answer"""
        
        visualizations = []
        
        # Quality score visualization
        if holistic_answer.quality_metrics:
            visualizations.append({
                "type": "quality_chart",
                "title": "Quality Metrics",
                "data": holistic_answer.quality_metrics,
                "chart_type": "radar",
                "styling": {"theme": "professional"}
            })
        
        # Confidence score gauge
        visualizations.append({
            "type": "confidence_gauge",
            "title": "Confidence Score",
            "data": {"score": holistic_answer.confidence_score},
            "chart_type": "gauge",
            "styling": {"color_scheme": "green_to_red"}
        })
        
        return visualizations
    
    def _generate_appendices(
        self,
        holistic_answer: HolisticAnswer,
        formatted_answer: FormattedAnswer,
        quality_report: QualityReport,
        context: StructuringContext
    ) -> List[Dict[str, Any]]:
        """Generate appendices with supporting information"""
        
        appendices = []
        
        # Quality report appendix
        appendices.append({
            "id": "quality_report",
            "title": "Quality Assessment Report",
            "content": {
                "overall_score": quality_report.overall_score,
                "individual_scores": quality_report.individual_scores,
                "validation_summary": quality_report.validation_summary
            },
            "type": "quality_data"
        })
        
        # Methodology appendix
        appendices.append({
            "id": "methodology",
            "title": "Analysis Methodology",
            "content": {
                "synthesis_strategy": "integrated",
                "validation_approach": "comprehensive",
                "personalization_level": "advanced"
            },
            "type": "methodology"
        })
        
        return appendices
    
    def _compile_citations(
        self,
        holistic_answer: HolisticAnswer,
        formatted_answer: FormattedAnswer,
        quality_report: QualityReport
    ) -> List[str]:
        """Compile citations from all sources"""
        
        citations = []
        
        # Add citations from formatted answer
        if hasattr(formatted_answer, 'citations'):
            citations.extend(formatted_answer.citations)
        
        # Add system citations
        citations.extend([
            "CherryAI Holistic Answer Synthesis Engine v3.1.0",
            "CherryAI Domain-Specific Answer Formatter v3.2.0",
            "CherryAI User-Personalized Result Optimizer v3.3.0",
            "CherryAI Answer Quality Validator v3.4.0",
            "CherryAI Final Answer Structuring v3.5.0"
        ])
        
        return citations
    
    def _apply_final_formatting(
        self,
        final_answer: FinalStructuredAnswer,
        context: StructuringContext
    ) -> FinalStructuredAnswer:
        """Apply final formatting based on export format and presentation mode"""
        
        # Apply format-specific styling
        if context.export_format == ExportFormat.MARKDOWN:
            final_answer = self._apply_markdown_formatting(final_answer)
        elif context.export_format == ExportFormat.HTML:
            final_answer = self._apply_html_formatting(final_answer)
        elif context.export_format == ExportFormat.JSON:
            final_answer = self._apply_json_formatting(final_answer)
        
        # Apply presentation mode styling
        if context.presentation_mode == PresentationMode.MOBILE_OPTIMIZED:
            final_answer = self._apply_mobile_optimization(final_answer)
        elif context.presentation_mode == PresentationMode.PRINT_FRIENDLY:
            final_answer = self._apply_print_optimization(final_answer)
        
        return final_answer
    
    def _apply_markdown_formatting(self, answer: FinalStructuredAnswer) -> FinalStructuredAnswer:
        """Apply markdown-specific formatting"""
        
        # Format sections with markdown headers
        for section in answer.main_sections:
            section.content = f"## {section.title}\n\n{section.content}"
        
        return answer
    
    def _apply_html_formatting(self, answer: FinalStructuredAnswer) -> FinalStructuredAnswer:
        """Apply HTML-specific formatting"""
        
        # Add HTML structure
        for section in answer.main_sections:
            section.content = f"<section><h2>{section.title}</h2><div>{section.content}</div></section>"
        
        return answer
    
    def _apply_json_formatting(self, answer: FinalStructuredAnswer) -> FinalStructuredAnswer:
        """Apply JSON-specific formatting"""
        
        # Ensure all content is JSON-serializable
        for section in answer.main_sections:
            if not isinstance(section.content, (str, dict, list)):
                section.content = str(section.content)
        
        return answer
    
    def _apply_mobile_optimization(self, answer: FinalStructuredAnswer) -> FinalStructuredAnswer:
        """Apply mobile-specific optimizations"""
        
        # Simplify content for mobile
        for section in answer.main_sections:
            if len(section.content) > 300:
                section.content = section.content[:300] + "..."
        
        return answer
    
    def _apply_print_optimization(self, answer: FinalStructuredAnswer) -> FinalStructuredAnswer:
        """Apply print-specific optimizations"""
        
        # Remove interactive elements for print
        answer.interactive_elements = []
        
        return answer
    
    def export_final_answer(
        self,
        final_answer: FinalStructuredAnswer,
        export_format: ExportFormat,
        output_path: str = None
    ) -> Union[str, Dict[str, Any]]:
        """Export final answer to specified format"""
        
        try:
            if export_format == ExportFormat.JSON:
                return self._export_to_json(final_answer)
            elif export_format == ExportFormat.MARKDOWN:
                return self._export_to_markdown(final_answer)
            elif export_format == ExportFormat.HTML:
                return self._export_to_html(final_answer)
            elif export_format == ExportFormat.PLAIN_TEXT:
                return self._export_to_plain_text(final_answer)
            else:
                return self._export_to_structured_data(final_answer)
                
        except Exception as e:
            logger.error(f"Export error: {str(e)}")
            raise
    
    def _export_to_json(self, answer: FinalStructuredAnswer) -> Dict[str, Any]:
        """Export answer to JSON format"""
        
        return {
            "answer_id": answer.answer_id,
            "title": answer.title,
            "executive_summary": answer.executive_summary,
            "sections": [
                {
                    "id": section.section_id,
                    "title": section.title,
                    "content": section.content,
                    "type": section.section_type,
                    "priority": section.priority
                }
                for section in answer.main_sections
            ],
            "quality_metrics": {
                "overall_score": answer.quality_report.overall_score,
                "quality_level": answer.quality_report.overall_level.value,
                "passed_validation": answer.quality_report.passed_validation
            },
            "metadata": {
                "generated_at": answer.generated_at.isoformat(),
                "structure_type": answer.structure_type.value,
                "export_format": answer.export_format.value,
                "presentation_mode": answer.presentation_mode.value
            }
        }
    
    def _export_to_markdown(self, answer: FinalStructuredAnswer) -> str:
        """Export answer to Markdown format"""
        
        content = []
        
        # Title
        content.append(f"# {answer.title}")
        content.append("")
        
        # Executive Summary
        content.append("## Executive Summary")
        content.append(answer.executive_summary)
        content.append("")
        
        # Main Sections
        for section in answer.main_sections:
            content.append(f"## {section.title}")
            content.append(section.content)
            content.append("")
        
        # Quality Metrics
        content.append("## Quality Assessment")
        content.append(f"**Overall Score:** {answer.quality_report.overall_score:.1f}/1.0")
        content.append(f"**Quality Level:** {answer.quality_report.overall_level.value}")
        content.append("")
        
        # Metadata
        content.append("---")
        content.append(f"*Generated: {answer.generated_at.strftime('%Y-%m-%d %H:%M:%S')}*")
        content.append(f"*Structure Type: {answer.structure_type.value}*")
        
        return "\n".join(content)
    
    def _export_to_html(self, answer: FinalStructuredAnswer) -> str:
        """Export answer to HTML format"""
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{answer.title}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .executive-summary {{ background-color: #f5f5f5; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                .quality-metrics {{ background-color: #e8f5e8; padding: 15px; border-radius: 5px; }}
                .metadata {{ color: #666; font-size: 0.9em; margin-top: 30px; }}
            </style>
        </head>
        <body>
            <h1>{answer.title}</h1>
            
            <div class="executive-summary">
                <h2>Executive Summary</h2>
                <p>{answer.executive_summary}</p>
            </div>
            
            {self._generate_html_sections(answer.main_sections)}
            
            <div class="quality-metrics">
                <h2>Quality Assessment</h2>
                <p><strong>Overall Score:</strong> {answer.quality_report.overall_score:.1f}/1.0</p>
                <p><strong>Quality Level:</strong> {answer.quality_report.overall_level.value}</p>
            </div>
            
            <div class="metadata">
                <p>Generated: {answer.generated_at.strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Structure Type: {answer.structure_type.value}</p>
            </div>
        </body>
        </html>
        """
        
        return html_content
    
    def _generate_html_sections(self, sections: List[AnswerSection]) -> str:
        """Generate HTML for main sections"""
        
        html_sections = []
        
        for section in sections:
            html_sections.append(f"""
            <div class="section">
                <h2>{section.title}</h2>
                <div>{section.content}</div>
            </div>
            """)
        
        return "\n".join(html_sections)
    
    def _export_to_plain_text(self, answer: FinalStructuredAnswer) -> str:
        """Export answer to plain text format"""
        
        content = []
        
        # Title
        content.append(answer.title)
        content.append("=" * len(answer.title))
        content.append("")
        
        # Executive Summary
        content.append("EXECUTIVE SUMMARY")
        content.append("-" * 17)
        content.append(answer.executive_summary)
        content.append("")
        
        # Main Sections
        for section in answer.main_sections:
            content.append(section.title.upper())
            content.append("-" * len(section.title))
            content.append(section.content)
            content.append("")
        
        # Quality Metrics
        content.append("QUALITY ASSESSMENT")
        content.append("-" * 18)
        content.append(f"Overall Score: {answer.quality_report.overall_score:.1f}/1.0")
        content.append(f"Quality Level: {answer.quality_report.overall_level.value}")
        content.append("")
        
        # Metadata
        content.append(f"Generated: {answer.generated_at.strftime('%Y-%m-%d %H:%M:%S')}")
        content.append(f"Structure Type: {answer.structure_type.value}")
        
        return "\n".join(content)
    
    def _export_to_structured_data(self, answer: FinalStructuredAnswer) -> Dict[str, Any]:
        """Export answer to structured data format"""
        
        return {
            "header": {
                "answer_id": answer.answer_id,
                "title": answer.title,
                "generated_at": answer.generated_at.isoformat(),
                "structure_type": answer.structure_type.value
            },
            "executive_summary": answer.executive_summary,
            "content": {
                "sections": [
                    {
                        "id": section.section_id,
                        "title": section.title,
                        "content": section.content,
                        "metadata": section.metadata
                    }
                    for section in answer.main_sections
                ],
                "components": [
                    {
                        "id": comp.component_id,
                        "type": comp.component_type,
                        "title": comp.title,
                        "content": comp.content
                    }
                    for comp in answer.components
                ]
            },
            "quality": {
                "overall_score": answer.quality_report.overall_score,
                "quality_level": answer.quality_report.overall_level.value,
                "validation_passed": answer.quality_report.passed_validation,
                "individual_scores": {
                    metric.value: score.score 
                    for metric, score in answer.quality_report.individual_scores.items()
                }
            },
            "metadata": {
                "processing_metadata": answer.metadata.__dict__,
                "source_components": {
                    "holistic_answer_id": answer.holistic_answer.answer_id,
                    "formatted_answer_format": answer.formatted_answer.format_type.value,
                    "optimization_score": answer.optimized_result.optimization_score
                }
            }
        }
    
    def create_structuring_context(
        self,
        structure_type: StructureType = StructureType.COMPREHENSIVE,
        export_format: ExportFormat = ExportFormat.MARKDOWN,
        presentation_mode: PresentationMode = PresentationMode.STATIC,
        target_audience: str = "general",
        use_case: str = "analysis"
    ) -> StructuringContext:
        """Create a structuring context with default settings"""
        
        return StructuringContext(
            structure_type=structure_type,
            export_format=export_format,
            presentation_mode=presentation_mode,
            target_audience=target_audience,
            use_case=use_case
        )
    
    def get_structuring_history(self) -> List[FinalStructuredAnswer]:
        """Get history of structured answers"""
        return self.structuring_history.copy()
    
    def get_supported_formats(self) -> Dict[str, List[str]]:
        """Get supported structure types, export formats, and presentation modes"""
        return {
            "structure_types": [st.value for st in StructureType],
            "export_formats": [ef.value for ef in ExportFormat],
            "presentation_modes": [pm.value for pm in PresentationMode]
        } 