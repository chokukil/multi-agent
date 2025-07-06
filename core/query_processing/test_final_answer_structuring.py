"""
Test suite for Final Answer Structuring Module
"""

import pytest
from datetime import datetime
from typing import Dict, List

from .final_answer_structuring import (
    FinalAnswerStructuring, StructureType, ExportFormat, PresentationMode,
    AnswerMetadata, StructuringContext, AnswerSection, AnswerComponent,
    FinalStructuredAnswer
)
from .holistic_answer_synthesis_engine import (
    HolisticAnswer, AnswerSection as HolisticAnswerSection, AnswerStyle
)
from .domain_specific_answer_formatter import (
    FormattedAnswer, OutputFormat, DomainType, FormattingStyle
)
from .user_personalized_result_optimizer import (
    OptimizedResult, PersonalizationLevel, UserRole, OptimizationStrategy,
    UserProfile, OptimizationContext, PersonalizationInsights
)
from .answer_quality_validator import (
    QualityReport, QualityLevel, QualityMetric, QualityScore
)


class TestFinalAnswerStructuring:
    """Test suite for FinalAnswerStructuring"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.structuring_engine = FinalAnswerStructuring()
        self.sample_holistic_answer = self.create_sample_holistic_answer()
        self.sample_formatted_answer = self.create_sample_formatted_answer()
        self.sample_optimized_result = self.create_sample_optimized_result()
        self.sample_quality_report = self.create_sample_quality_report()
        self.sample_structuring_context = self.create_sample_structuring_context()
    
    def create_sample_holistic_answer(self) -> HolisticAnswer:
        """Create sample holistic answer for testing"""
        
        sections = [
            HolisticAnswerSection(
                title="Market Analysis",
                content="Comprehensive market analysis showing strong growth trends and emerging opportunities in the digital transformation sector.",
                priority=1,
                section_type="analysis",
                confidence=0.88
            ),
            HolisticAnswerSection(
                title="Strategic Recommendations",
                content="Based on the analysis, we recommend focusing on three key areas: technology adoption, market expansion, and strategic partnerships.",
                priority=2,
                section_type="recommendations",
                confidence=0.92
            )
        ]
        
        return HolisticAnswer(
            answer_id="test_holistic_001",
            query_summary="Market analysis and strategic recommendations for digital transformation",
            executive_summary="This analysis provides comprehensive insights into market trends and strategic recommendations for successful digital transformation initiatives.",
            main_sections=sections,
            key_insights=[
                "Digital transformation market is experiencing 23% YoY growth",
                "Customer expectations are driving technology adoption",
                "Strategic partnerships are critical for success"
            ],
            recommendations=[
                "Invest in cloud infrastructure and AI capabilities",
                "Develop customer-centric digital experiences",
                "Form strategic partnerships with technology providers"
            ],
            next_steps=[
                "Conduct detailed technology assessment",
                "Develop 18-month implementation roadmap",
                "Establish governance framework for digital initiatives"
            ],
            confidence_score=0.89,
            quality_metrics={
                "completeness": 0.91,
                "accuracy": 0.87,
                "relevance": 0.93
            },
            synthesis_metadata={
                "analysis_depth": 3,
                "insight_count": 3,
                "recommendation_count": 3
            },
            generated_at=datetime.now(),
            synthesis_time=3.2
        )
    
    def create_sample_formatted_answer(self) -> FormattedAnswer:
        """Create sample formatted answer for testing"""
        
        return FormattedAnswer(
            content="# Digital Transformation Analysis\n\n## Executive Summary\n\nThis analysis provides comprehensive insights into market trends and strategic recommendations for digital transformation.\n\n## Key Findings\n\n- Market growth of 23% YoY\n- Customer-driven adoption\n- Partnership importance\n\n## Recommendations\n\n1. Invest in cloud and AI\n2. Focus on customer experience\n3. Build strategic partnerships",
            format_type=OutputFormat.MARKDOWN,
            domain=DomainType.BUSINESS,
            style=FormattingStyle.EXECUTIVE,
            metadata={
                "word_count": 120,
                "sections": 3,
                "formatting_time": "2024-01-01T10:00:00Z"
            }
        )
    
    def create_sample_optimized_result(self) -> OptimizedResult:
        """Create sample optimized result for testing"""
        
        # Create required nested objects
        user_profile = UserProfile(
            user_id="test_user",
            role=UserRole.EXECUTIVE,
            domain_expertise={"business": 0.9, "technology": 0.7},
            preferences={},
            interaction_history=[],
            learning_weights={},
            personalization_level=PersonalizationLevel.ADVANCED
        )
        
        optimization_context = OptimizationContext(
            user_profile=user_profile,
            current_query="Digital transformation analysis",
            domain_context=None,
            intent_analysis=None
        )
        
        personalization_insights = PersonalizationInsights(
            applied_optimizations=["executive_summary", "strategic_focus", "action_oriented"],
            preference_matches={"strategic_focus": 0.95, "executive_language": 0.88},
            learning_contributions={"interaction_history": 0.8},
            confidence_score=0.91,
            optimization_impact=0.87
        )
        
        return OptimizedResult(
            original_result=self.create_sample_formatted_answer(),
            optimized_content="**Executive-Focused Digital Transformation Analysis**\n\nThis strategic analysis provides actionable insights for executive decision-making in digital transformation initiatives, emphasizing ROI and competitive advantage.\n\n**Strategic Imperatives:**\n- 23% market growth creates urgency for immediate action\n- Customer expectations demand premium digital experiences\n- Strategic partnerships essential for competitive positioning\n\n**Executive Recommendations:**\n1. **Technology Investment**: Prioritize cloud infrastructure and AI capabilities for operational efficiency\n2. **Customer Experience**: Develop differentiated digital touchpoints to capture market share\n3. **Partnership Strategy**: Establish alliances with leading technology providers\n\n**Next Steps for Leadership:**\n- Approve technology assessment budget within 30 days\n- Assign dedicated transformation team with C-level sponsorship\n- Establish quarterly review process for tracking progress",
            personalization_metadata={
                "user_id": "test_user",
                "optimization_time": "2024-01-01T10:00:00Z",
                "applied_preferences": ["executive_summary", "strategic_focus", "action_oriented"]
            },
            optimization_context=optimization_context,
            personalization_insights=personalization_insights,
            optimization_score=0.91
        )
    
    def create_sample_quality_report(self) -> QualityReport:
        """Create sample quality report for testing"""
        
        individual_scores = {
            QualityMetric.ACCURACY: QualityScore(
                metric=QualityMetric.ACCURACY,
                score=0.87,
                level=QualityLevel.GOOD,
                explanation="High accuracy based on confidence scores and supporting evidence"
            ),
            QualityMetric.COMPLETENESS: QualityScore(
                metric=QualityMetric.COMPLETENESS,
                score=0.91,
                level=QualityLevel.EXCELLENT,
                explanation="Comprehensive coverage of all requested aspects"
            ),
            QualityMetric.RELEVANCE: QualityScore(
                metric=QualityMetric.RELEVANCE,
                score=0.93,
                level=QualityLevel.EXCELLENT,
                explanation="Highly relevant to user query and context"
            )
        }
        
        return QualityReport(
            overall_score=0.89,
            overall_level=QualityLevel.GOOD,
            individual_scores=individual_scores,
            validation_summary="This analysis meets high quality standards with excellent completeness and relevance, and good accuracy.",
            passed_validation=True,
            strengths=[
                "Comprehensive analysis coverage",
                "High relevance to business context",
                "Clear actionable recommendations"
            ],
            recommendations=[
                "Consider adding more quantitative metrics",
                "Include risk assessment section"
            ]
        )
    
    def create_sample_structuring_context(self) -> StructuringContext:
        """Create sample structuring context for testing"""
        
        return StructuringContext(
            structure_type=StructureType.COMPREHENSIVE,
            export_format=ExportFormat.MARKDOWN,
            presentation_mode=PresentationMode.STATIC,
            target_audience="executive",
            use_case="strategic_analysis"
        )
    
    def test_initialization(self):
        """Test structuring engine initialization"""
        
        engine = FinalAnswerStructuring()
        
        assert engine.structure_templates is not None
        assert engine.formatting_rules is not None
        assert engine.presentation_styles is not None
        assert len(engine.structuring_history) == 0
        assert StructureType.COMPREHENSIVE in engine.structure_templates
        assert ExportFormat.MARKDOWN in engine.formatting_rules
        assert PresentationMode.STATIC in engine.presentation_styles
    
    def test_structure_final_answer(self):
        """Test complete final answer structuring"""
        
        final_answer = self.structuring_engine.structure_final_answer(
            self.sample_holistic_answer,
            self.sample_formatted_answer,
            self.sample_optimized_result,
            self.sample_quality_report,
            self.sample_structuring_context
        )
        
        assert isinstance(final_answer, FinalStructuredAnswer)
        assert final_answer.answer_id is not None
        assert final_answer.title != ""
        assert final_answer.executive_summary != ""
        assert len(final_answer.main_sections) > 0
        assert len(final_answer.components) > 0
        assert final_answer.quality_report == self.sample_quality_report
        assert final_answer.structure_type == StructureType.COMPREHENSIVE
        assert final_answer.export_format == ExportFormat.MARKDOWN
        assert final_answer.presentation_mode == PresentationMode.STATIC
    
    def test_structure_types(self):
        """Test different structure types"""
        
        structure_types = [
            StructureType.EXECUTIVE_REPORT,
            StructureType.TECHNICAL_DOCUMENT,
            StructureType.PRESENTATION,
            StructureType.DASHBOARD,
            StructureType.COMPREHENSIVE
        ]
        
        for structure_type in structure_types:
            context = StructuringContext(
                structure_type=structure_type,
                export_format=ExportFormat.MARKDOWN,
                presentation_mode=PresentationMode.STATIC,
                target_audience="general",
                use_case="analysis"
            )
            
            final_answer = self.structuring_engine.structure_final_answer(
                self.sample_holistic_answer,
                self.sample_formatted_answer,
                self.sample_optimized_result,
                self.sample_quality_report,
                context
            )
            
            assert final_answer.structure_type == structure_type
            assert len(final_answer.main_sections) > 0
            assert final_answer.title != ""
    
    def test_export_formats(self):
        """Test different export formats"""
        
        export_formats = [
            ExportFormat.JSON,
            ExportFormat.MARKDOWN,
            ExportFormat.HTML,
            ExportFormat.PLAIN_TEXT,
            ExportFormat.STRUCTURED_DATA
        ]
        
        final_answer = self.structuring_engine.structure_final_answer(
            self.sample_holistic_answer,
            self.sample_formatted_answer,
            self.sample_optimized_result,
            self.sample_quality_report,
            self.sample_structuring_context
        )
        
        for export_format in export_formats:
            exported = self.structuring_engine.export_final_answer(
                final_answer,
                export_format
            )
            
            assert exported is not None
            
            if export_format == ExportFormat.JSON:
                assert isinstance(exported, dict)
                assert "answer_id" in exported
                assert "title" in exported
            elif export_format in [ExportFormat.MARKDOWN, ExportFormat.HTML, ExportFormat.PLAIN_TEXT]:
                assert isinstance(exported, str)
                assert len(exported) > 0
    
    def test_presentation_modes(self):
        """Test different presentation modes"""
        
        presentation_modes = [
            PresentationMode.INTERACTIVE,
            PresentationMode.STATIC,
            PresentationMode.MOBILE_OPTIMIZED,
            PresentationMode.PRINT_FRIENDLY
        ]
        
        for presentation_mode in presentation_modes:
            context = StructuringContext(
                structure_type=StructureType.COMPREHENSIVE,
                export_format=ExportFormat.MARKDOWN,
                presentation_mode=presentation_mode,
                target_audience="general",
                use_case="analysis"
            )
            
            final_answer = self.structuring_engine.structure_final_answer(
                self.sample_holistic_answer,
                self.sample_formatted_answer,
                self.sample_optimized_result,
                self.sample_quality_report,
                context
            )
            
            assert final_answer.presentation_mode == presentation_mode
            
            if presentation_mode == PresentationMode.INTERACTIVE:
                assert len(final_answer.interactive_elements) > 0
            elif presentation_mode == PresentationMode.PRINT_FRIENDLY:
                assert len(final_answer.interactive_elements) == 0
    
    def test_answer_metadata_creation(self):
        """Test answer metadata creation"""
        
        final_answer = self.structuring_engine.structure_final_answer(
            self.sample_holistic_answer,
            self.sample_formatted_answer,
            self.sample_optimized_result,
            self.sample_quality_report,
            self.sample_structuring_context,
            user_id="test_user_123",
            session_id="test_session_456"
        )
        
        metadata = final_answer.metadata
        assert isinstance(metadata, AnswerMetadata)
        assert metadata.user_id == "test_user_123"
        assert metadata.session_id == "test_session_456"
        assert metadata.answer_id == final_answer.answer_id
        assert metadata.query_hash is not None
        assert "holistic_synthesis" in metadata.component_versions
        assert "overall_score" in metadata.quality_metrics
    
    def test_section_content_generation(self):
        """Test section content generation"""
        
        final_answer = self.structuring_engine.structure_final_answer(
            self.sample_holistic_answer,
            self.sample_formatted_answer,
            self.sample_optimized_result,
            self.sample_quality_report,
            self.sample_structuring_context
        )
        
        # Check that sections contain appropriate content
        section_titles = [section.title for section in final_answer.main_sections]
        assert "Executive Summary" in section_titles
        assert "Key Findings" in section_titles or "Detailed Analysis" in section_titles
        assert "Recommendations" in section_titles
        
        # Check section content is not empty
        for section in final_answer.main_sections:
            assert section.content != ""
            assert section.section_id != ""
            assert section.priority > 0
    
    def test_component_creation(self):
        """Test answer component creation"""
        
        final_answer = self.structuring_engine.structure_final_answer(
            self.sample_holistic_answer,
            self.sample_formatted_answer,
            self.sample_optimized_result,
            self.sample_quality_report,
            self.sample_structuring_context
        )
        
        assert len(final_answer.components) >= 3  # At least main_content, quality_metrics, insights
        
        component_types = [comp.component_type for comp in final_answer.components]
        assert "content" in component_types
        assert "metrics" in component_types
        assert "insights" in component_types
        
        for component in final_answer.components:
            assert component.component_id != ""
            assert component.title != ""
            assert component.content is not None
    
    def test_interactive_elements_generation(self):
        """Test interactive elements generation"""
        
        interactive_context = StructuringContext(
            structure_type=StructureType.COMPREHENSIVE,
            export_format=ExportFormat.HTML,
            presentation_mode=PresentationMode.INTERACTIVE,
            target_audience="general",
            use_case="analysis"
        )
        
        final_answer = self.structuring_engine.structure_final_answer(
            self.sample_holistic_answer,
            self.sample_formatted_answer,
            self.sample_optimized_result,
            self.sample_quality_report,
            interactive_context
        )
        
        assert len(final_answer.interactive_elements) > 0
        
        for element in final_answer.interactive_elements:
            assert "type" in element
            assert "title" in element
            assert element["type"] in ["expandable_section", "interactive_insights", "action_buttons"]
    
    def test_visualizations_creation(self):
        """Test visualizations creation"""
        
        final_answer = self.structuring_engine.structure_final_answer(
            self.sample_holistic_answer,
            self.sample_formatted_answer,
            self.sample_optimized_result,
            self.sample_quality_report,
            self.sample_structuring_context
        )
        
        assert len(final_answer.visualizations) > 0
        
        for visualization in final_answer.visualizations:
            assert "type" in visualization
            assert "title" in visualization
            assert "data" in visualization
            assert visualization["type"] in ["quality_chart", "confidence_gauge"]
    
    def test_appendices_generation(self):
        """Test appendices generation"""
        
        final_answer = self.structuring_engine.structure_final_answer(
            self.sample_holistic_answer,
            self.sample_formatted_answer,
            self.sample_optimized_result,
            self.sample_quality_report,
            self.sample_structuring_context
        )
        
        assert len(final_answer.appendices) > 0
        
        appendix_ids = [appendix["id"] for appendix in final_answer.appendices]
        assert "quality_report" in appendix_ids
        assert "methodology" in appendix_ids
        
        for appendix in final_answer.appendices:
            assert "id" in appendix
            assert "title" in appendix
            assert "content" in appendix
            assert "type" in appendix
    
    def test_citations_compilation(self):
        """Test citations compilation"""
        
        final_answer = self.structuring_engine.structure_final_answer(
            self.sample_holistic_answer,
            self.sample_formatted_answer,
            self.sample_optimized_result,
            self.sample_quality_report,
            self.sample_structuring_context
        )
        
        assert len(final_answer.citations) > 0
        
        # Check that system citations are included
        citation_text = " ".join(final_answer.citations)
        assert "CherryAI" in citation_text
        assert "v3.1.0" in citation_text or "v3.2.0" in citation_text
    
    def test_json_export(self):
        """Test JSON export functionality"""
        
        final_answer = self.structuring_engine.structure_final_answer(
            self.sample_holistic_answer,
            self.sample_formatted_answer,
            self.sample_optimized_result,
            self.sample_quality_report,
            self.sample_structuring_context
        )
        
        json_export = self.structuring_engine.export_final_answer(
            final_answer,
            ExportFormat.JSON
        )
        
        assert isinstance(json_export, dict)
        assert "answer_id" in json_export
        assert "title" in json_export
        assert "executive_summary" in json_export
        assert "sections" in json_export
        assert "quality_metrics" in json_export
        assert "metadata" in json_export
        
        # Check sections structure
        assert len(json_export["sections"]) > 0
        for section in json_export["sections"]:
            assert "id" in section
            assert "title" in section
            assert "content" in section
            assert "priority" in section
    
    def test_markdown_export(self):
        """Test Markdown export functionality"""
        
        final_answer = self.structuring_engine.structure_final_answer(
            self.sample_holistic_answer,
            self.sample_formatted_answer,
            self.sample_optimized_result,
            self.sample_quality_report,
            self.sample_structuring_context
        )
        
        markdown_export = self.structuring_engine.export_final_answer(
            final_answer,
            ExportFormat.MARKDOWN
        )
        
        assert isinstance(markdown_export, str)
        assert len(markdown_export) > 0
        
        # Check markdown formatting
        assert markdown_export.startswith("# ")  # Title
        assert "## Executive Summary" in markdown_export
        assert "## Quality Assessment" in markdown_export
        assert "**Overall Score:**" in markdown_export
        assert "*Generated:" in markdown_export
    
    def test_html_export(self):
        """Test HTML export functionality"""
        
        final_answer = self.structuring_engine.structure_final_answer(
            self.sample_holistic_answer,
            self.sample_formatted_answer,
            self.sample_optimized_result,
            self.sample_quality_report,
            self.sample_structuring_context
        )
        
        html_export = self.structuring_engine.export_final_answer(
            final_answer,
            ExportFormat.HTML
        )
        
        assert isinstance(html_export, str)
        assert len(html_export) > 0
        
        # Check HTML structure
        assert "<!DOCTYPE html>" in html_export
        assert "<html>" in html_export
        assert "<head>" in html_export
        assert "<body>" in html_export
        assert "<h1>" in html_export
        assert "<h2>" in html_export
        assert "executive-summary" in html_export
        assert "quality-metrics" in html_export
    
    def test_plain_text_export(self):
        """Test plain text export functionality"""
        
        final_answer = self.structuring_engine.structure_final_answer(
            self.sample_holistic_answer,
            self.sample_formatted_answer,
            self.sample_optimized_result,
            self.sample_quality_report,
            self.sample_structuring_context
        )
        
        text_export = self.structuring_engine.export_final_answer(
            final_answer,
            ExportFormat.PLAIN_TEXT
        )
        
        assert isinstance(text_export, str)
        assert len(text_export) > 0
        
        # Check plain text formatting
        assert "EXECUTIVE SUMMARY" in text_export
        assert "QUALITY ASSESSMENT" in text_export
        assert "Overall Score:" in text_export
        assert "Generated:" in text_export
        assert "=" in text_export  # Title underline
        assert "-" in text_export  # Section underlines
    
    def test_structured_data_export(self):
        """Test structured data export functionality"""
        
        final_answer = self.structuring_engine.structure_final_answer(
            self.sample_holistic_answer,
            self.sample_formatted_answer,
            self.sample_optimized_result,
            self.sample_quality_report,
            self.sample_structuring_context
        )
        
        structured_export = self.structuring_engine.export_final_answer(
            final_answer,
            ExportFormat.STRUCTURED_DATA
        )
        
        assert isinstance(structured_export, dict)
        assert "header" in structured_export
        assert "executive_summary" in structured_export
        assert "content" in structured_export
        assert "quality" in structured_export
        assert "metadata" in structured_export
        
        # Check header structure
        header = structured_export["header"]
        assert "answer_id" in header
        assert "title" in header
        assert "generated_at" in header
        assert "structure_type" in header
        
        # Check content structure
        content = structured_export["content"]
        assert "sections" in content
        assert "components" in content
        assert len(content["sections"]) > 0
        assert len(content["components"]) > 0
    
    def test_create_structuring_context(self):
        """Test structuring context creation"""
        
        context = self.structuring_engine.create_structuring_context(
            structure_type=StructureType.EXECUTIVE_REPORT,
            export_format=ExportFormat.HTML,
            presentation_mode=PresentationMode.INTERACTIVE,
            target_audience="executives",
            use_case="strategic_planning"
        )
        
        assert isinstance(context, StructuringContext)
        assert context.structure_type == StructureType.EXECUTIVE_REPORT
        assert context.export_format == ExportFormat.HTML
        assert context.presentation_mode == PresentationMode.INTERACTIVE
        assert context.target_audience == "executives"
        assert context.use_case == "strategic_planning"
    
    def test_structuring_history(self):
        """Test structuring history tracking"""
        
        # Create multiple final answers
        for i in range(3):
            self.structuring_engine.structure_final_answer(
                self.sample_holistic_answer,
                self.sample_formatted_answer,
                self.sample_optimized_result,
                self.sample_quality_report,
                self.sample_structuring_context
            )
        
        history = self.structuring_engine.get_structuring_history()
        
        assert len(history) == 3
        assert all(isinstance(answer, FinalStructuredAnswer) for answer in history)
        assert all(answer.answer_id is not None for answer in history)
    
    def test_supported_formats(self):
        """Test supported formats retrieval"""
        
        supported_formats = self.structuring_engine.get_supported_formats()
        
        assert isinstance(supported_formats, dict)
        assert "structure_types" in supported_formats
        assert "export_formats" in supported_formats
        assert "presentation_modes" in supported_formats
        
        assert "comprehensive" in supported_formats["structure_types"]
        assert "markdown" in supported_formats["export_formats"]
        assert "static" in supported_formats["presentation_modes"]
    
    def test_answer_title_generation(self):
        """Test answer title generation for different structure types"""
        
        structure_types = [
            (StructureType.EXECUTIVE_REPORT, "Executive Analysis:"),
            (StructureType.TECHNICAL_DOCUMENT, "Technical Documentation:"),
            (StructureType.PRESENTATION, "Presentation:"),
            (StructureType.DASHBOARD, "Analytics Dashboard:"),
            (StructureType.COMPREHENSIVE, "Comprehensive Analysis:")
        ]
        
        for structure_type, expected_prefix in structure_types:
            context = StructuringContext(
                structure_type=structure_type,
                export_format=ExportFormat.MARKDOWN,
                presentation_mode=PresentationMode.STATIC,
                target_audience="general",
                use_case="analysis"
            )
            
            final_answer = self.structuring_engine.structure_final_answer(
                self.sample_holistic_answer,
                self.sample_formatted_answer,
                self.sample_optimized_result,
                self.sample_quality_report,
                context
            )
            
            assert final_answer.title.startswith(expected_prefix)
    
    def test_executive_summary_enhancement(self):
        """Test executive summary enhancement with quality and personalization info"""
        
        final_answer = self.structuring_engine.structure_final_answer(
            self.sample_holistic_answer,
            self.sample_formatted_answer,
            self.sample_optimized_result,
            self.sample_quality_report,
            self.sample_structuring_context
        )
        
        # Check that executive summary includes quality indicator
        assert "Quality Score:" in final_answer.executive_summary
        assert "personalized" in final_answer.executive_summary.lower()
        
        # Check that it starts with the original summary
        assert final_answer.executive_summary.startswith(
            self.sample_holistic_answer.executive_summary
        )
    
    def test_mobile_optimization(self):
        """Test mobile optimization features"""
        
        mobile_context = StructuringContext(
            structure_type=StructureType.COMPREHENSIVE,
            export_format=ExportFormat.HTML,
            presentation_mode=PresentationMode.MOBILE_OPTIMIZED,
            target_audience="general",
            use_case="analysis"
        )
        
        final_answer = self.structuring_engine.structure_final_answer(
            self.sample_holistic_answer,
            self.sample_formatted_answer,
            self.sample_optimized_result,
            self.sample_quality_report,
            mobile_context
        )
        
        assert final_answer.presentation_mode == PresentationMode.MOBILE_OPTIMIZED
        
        # Check that content is optimized for mobile (shortened)
        for section in final_answer.main_sections:
            if len(section.content) > 300:
                assert section.content.endswith("...")
    
    def test_error_handling(self):
        """Test error handling in structuring process"""
        
        # Test with invalid context
        invalid_context = StructuringContext(
            structure_type=StructureType.COMPREHENSIVE,
            export_format=ExportFormat.MARKDOWN,
            presentation_mode=PresentationMode.STATIC,
            target_audience="",
            use_case=""
        )
        
        # This should still work despite empty target_audience and use_case
        final_answer = self.structuring_engine.structure_final_answer(
            self.sample_holistic_answer,
            self.sample_formatted_answer,
            self.sample_optimized_result,
            self.sample_quality_report,
            invalid_context
        )
        
        assert isinstance(final_answer, FinalStructuredAnswer)
        assert final_answer.answer_id is not None
    
    def test_comprehensive_integration(self):
        """Test comprehensive integration of all Phase 3 components"""
        
        final_answer = self.structuring_engine.structure_final_answer(
            self.sample_holistic_answer,
            self.sample_formatted_answer,
            self.sample_optimized_result,
            self.sample_quality_report,
            self.sample_structuring_context
        )
        
        # Verify all source components are preserved
        assert final_answer.holistic_answer == self.sample_holistic_answer
        assert final_answer.formatted_answer == self.sample_formatted_answer
        assert final_answer.optimized_result == self.sample_optimized_result
        assert final_answer.quality_report == self.sample_quality_report
        
        # Verify integration quality
        assert final_answer.quality_report.overall_score > 0.8  # Should be high quality
        assert len(final_answer.main_sections) >= 4  # Should have comprehensive sections
        assert len(final_answer.components) >= 3  # Should have multiple components
        assert len(final_answer.visualizations) >= 2  # Should have visualizations
        assert len(final_answer.appendices) >= 2  # Should have appendices
        assert len(final_answer.citations) >= 5  # Should have citations
        
        # Verify metadata completeness
        assert final_answer.metadata.component_versions["holistic_synthesis"] == "3.1.0"
        assert final_answer.metadata.component_versions["domain_formatting"] == "3.2.0"
        assert final_answer.metadata.component_versions["personalization"] == "3.3.0"
        assert final_answer.metadata.component_versions["quality_validation"] == "3.4.0"
        assert final_answer.metadata.component_versions["final_structuring"] == "3.5.0" 