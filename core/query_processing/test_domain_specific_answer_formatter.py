"""
Test suite for Domain-Specific Answer Formatter Module

This module contains comprehensive tests for the domain-specific answer formatting
functionality, ensuring proper formatting across different domains and output formats.

Author: CherryAI Development Team
Version: 1.0.0
"""

import pytest
from datetime import datetime
from typing import Dict, List, Any

from .domain_specific_answer_formatter import (
    DomainSpecificAnswerFormatter,
    DomainType,
    OutputFormat,
    FormattingStyle,
    FormattingContext,
    DomainFormattingRules,
    FormattedAnswer
)
from .holistic_answer_synthesis_engine import (
    HolisticAnswer,
    AnswerSection,
    AnswerStyle
)
from .intent_analyzer import (
    DetailedIntentAnalysis,
    PerspectiveAnalysis,
    AnalysisPerspective,
    QueryComplexity,
    UrgencyLevel
)
from .domain_extractor import (
    EnhancedDomainKnowledge,
    DomainTaxonomy,
    KnowledgeItem,
    MethodologyMap,
    RiskAssessment,
    KnowledgeConfidence,
    KnowledgeSource
)


class TestDomainSpecificAnswerFormatter:
    """Test cases for Domain-Specific Answer Formatter"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.formatter = DomainSpecificAnswerFormatter()
        self.sample_holistic_answer = self._create_sample_holistic_answer()
        self.sample_domain_knowledge = self._create_sample_domain_knowledge()
        self.sample_intent_analysis = self._create_sample_intent_analysis()
    
    def _create_sample_holistic_answer(self) -> HolisticAnswer:
        """Create a sample holistic answer for testing"""
        from datetime import datetime
        
        return HolisticAnswer(
            answer_id="test_answer_001",
            query_summary="How can we optimize our data processing pipeline?",
            executive_summary="The data processing pipeline can be optimized through several key strategies including architectural improvements, performance tuning, and resource optimization. Current bottlenecks exist in data ingestion and transformation phases.",
            main_sections=[
                AnswerSection(
                    title="Performance Analysis",
                    content="Current system shows latency issues in data processing with average response times of 2.5 seconds. Memory usage peaks at 85% during peak hours.",
                    priority=1,
                    section_type="analysis",
                    confidence=0.9
                ),
                AnswerSection(
                    title="Technical Recommendations",
                    content="Implement parallel processing, optimize database queries, and consider implementing caching mechanisms to reduce load times.",
                    priority=2,
                    section_type="recommendations",
                    confidence=0.8
                ),
                AnswerSection(
                    title="Implementation Strategy",
                    content="Phase 1: Database optimization (Week 1-2), Phase 2: Parallel processing implementation (Week 3-4), Phase 3: Caching layer (Week 5-6)",
                    priority=3,
                    section_type="implementation",
                    confidence=0.7
                )
            ],
            key_insights=[
                "Data processing bottlenecks occur during peak hours",
                "Memory optimization could improve performance by 40%",
                "Parallel processing implementation is critical for scalability"
            ],
            recommendations=[
                "Implement database indexing for frequently queried fields",
                "Deploy caching layer for repeated operations",
                "Optimize memory allocation for data processing tasks"
            ],
            next_steps=[
                "Conduct detailed performance profiling",
                "Design parallel processing architecture",
                "Implement proof of concept for caching solution"
            ],
            confidence_score=0.85,
            quality_metrics={
                "completeness": 0.9,
                "accuracy": 0.8,
                "relevance": 0.9,
                "actionability": 0.85
            },
            synthesis_metadata={
                "test_mode": True,
                "creation_time": datetime.now().isoformat()
            },
            generated_at=datetime.now(),
            synthesis_time=1.5
        )
    
    def _create_sample_domain_knowledge(self) -> EnhancedDomainKnowledge:
        """Create sample domain knowledge for testing"""
        from .intelligent_query_processor import DomainType
        
        return EnhancedDomainKnowledge(
            taxonomy=DomainTaxonomy(
                primary_domain=DomainType.MANUFACTURING,
                sub_domains=["data_processing", "performance_optimization"],
                industry_sector="technology",
                business_function="data_engineering",
                technical_area="system_performance",
                confidence_score=0.85
            ),
            key_concepts={
                "data_pipeline": KnowledgeItem(
                    item="Data Pipeline",
                    confidence=KnowledgeConfidence.HIGH,
                    source=KnowledgeSource.EXPLICIT_MENTION,
                    explanation="A series of data processing steps",
                    related_items=["Performance", "Scalability", "Optimization"]
                ),
                "performance_optimization": KnowledgeItem(
                    item="Performance Optimization",
                    confidence=KnowledgeConfidence.HIGH,
                    source=KnowledgeSource.TECHNICAL_PATTERNS,
                    explanation="Techniques to improve system performance",
                    related_items=["Caching", "Parallel Processing", "Database Tuning"]
                )
            },
            technical_terms={
                "latency": KnowledgeItem(
                    item="Latency",
                    confidence=KnowledgeConfidence.HIGH,
                    source=KnowledgeSource.TECHNICAL_PATTERNS,
                    explanation="Time delay in data processing",
                    related_items=["Performance", "Response Time"]
                )
            },
            methodology_map=MethodologyMap(
                standard_methodologies=["Agile Development", "DevOps"],
                best_practices=["Performance Monitoring", "Load Testing"],
                tools_and_technologies=["Database Optimization", "Caching Systems"],
                quality_standards=["ISO 9001", "SLA Compliance"],
                compliance_requirements=["Data Protection", "Security Standards"]
            ),
            risk_assessment=RiskAssessment(
                technical_risks=["System Downtime", "Performance Degradation"],
                business_risks=["Revenue Loss", "Customer Dissatisfaction"],
                operational_risks=["Resource Constraints", "Skill Gaps"],
                compliance_risks=["Data Breach", "Regulatory Violations"],
                mitigation_strategies=["Redundancy", "Monitoring", "Training"]
            ),
            success_metrics=["Response Time", "Throughput", "Reliability"],
            stakeholder_map={
                "technical": ["DevOps Team", "Data Engineers"],
                "business": ["Product Managers", "Operations Team"]
            },
            business_context="High-performance data processing system optimization",
            extraction_confidence=0.85
        )
    
    def _create_sample_intent_analysis(self) -> DetailedIntentAnalysis:
        """Create sample intent analysis for testing"""
        from .intelligent_query_processor import QueryType
        
        return DetailedIntentAnalysis(
            primary_intent="technical_optimization",
            secondary_intents=["performance_improvement", "cost_reduction"],
            query_type=QueryType.ANALYTICAL,
            complexity_level=QueryComplexity.COMPLEX,
            urgency_level=UrgencyLevel.HIGH,
            perspectives={
                AnalysisPerspective.TECHNICAL_IMPLEMENTER: PerspectiveAnalysis(
                    perspective=AnalysisPerspective.TECHNICAL_IMPLEMENTER,
                    primary_concerns=["performance", "scalability", "optimization"],
                    methodology_suggestions=["Performance Profiling", "Database Optimization"],
                    potential_challenges=["Resource Constraints", "Complexity"],
                    success_criteria=["Improved Response Time", "Better Throughput"],
                    estimated_effort=0.8,
                    confidence_level=0.9
                ),
                AnalysisPerspective.BUSINESS_ANALYST: PerspectiveAnalysis(
                    perspective=AnalysisPerspective.BUSINESS_ANALYST,
                    primary_concerns=["efficiency", "cost_reduction", "productivity"],
                    methodology_suggestions=["ROI Analysis", "Cost-Benefit Analysis"],
                    potential_challenges=["Budget Constraints", "Timeline Pressure"],
                    success_criteria=["Cost Savings", "Productivity Gains"],
                    estimated_effort=0.6,
                    confidence_level=0.7
                )
            },
            overall_confidence=0.85,
            execution_priority=7,
            estimated_timeline="2-3 weeks",
            critical_dependencies=["technical_resources", "data_access"]
        )
    
    def test_formatter_initialization(self):
        """Test formatter initialization"""
        formatter = DomainSpecificAnswerFormatter()
        
        assert formatter is not None
        assert len(formatter.domain_rules) > 0
        assert len(formatter.formatting_templates) > 0
        assert len(formatter.terminology_maps) > 0
        
        # Check that all required domains are supported
        assert DomainType.TECHNICAL in formatter.domain_rules
        assert DomainType.BUSINESS in formatter.domain_rules
        assert DomainType.ANALYTICS in formatter.domain_rules
    
    def test_create_formatting_context(self):
        """Test formatting context creation"""
        context = self.formatter.create_formatting_context(
            domain_knowledge=self.sample_domain_knowledge,
            intent_analysis=self.sample_intent_analysis
        )
        
        assert context is not None
        assert context.domain in DomainType
        assert context.output_format in OutputFormat
        assert context.style in FormattingStyle
        assert context.target_audience is not None
        assert isinstance(context.priority_sections, list)
    
    def test_format_technical_answer(self):
        """Test formatting for technical domain"""
        context = FormattingContext(
            domain=DomainType.TECHNICAL,
            output_format=OutputFormat.TECHNICAL_REPORT,
            style=FormattingStyle.TECHNICAL,
            target_audience="technical_team"
        )
        
        formatted_answer = self.formatter.format_answer(
            holistic_answer=self.sample_holistic_answer,
            domain_knowledge=self.sample_domain_knowledge,
            intent_analysis=self.sample_intent_analysis,
            formatting_context=context
        )
        
        assert formatted_answer is not None
        assert formatted_answer.format_type == OutputFormat.TECHNICAL_REPORT
        assert formatted_answer.domain == DomainType.TECHNICAL
        assert formatted_answer.style == FormattingStyle.TECHNICAL
        assert len(formatted_answer.content) > 0
        assert len(formatted_answer.sections) > 0
    
    def test_format_business_answer(self):
        """Test formatting for business domain"""
        context = FormattingContext(
            domain=DomainType.BUSINESS,
            output_format=OutputFormat.EXECUTIVE_SUMMARY,
            style=FormattingStyle.EXECUTIVE,
            target_audience="executives"
        )
        
        formatted_answer = self.formatter.format_answer(
            holistic_answer=self.sample_holistic_answer,
            domain_knowledge=self.sample_domain_knowledge,
            intent_analysis=self.sample_intent_analysis,
            formatting_context=context
        )
        
        assert formatted_answer is not None
        assert formatted_answer.format_type == OutputFormat.EXECUTIVE_SUMMARY
        assert formatted_answer.domain == DomainType.BUSINESS
        assert "Executive Summary" in formatted_answer.content or "executive_summary" in formatted_answer.content.lower()
        assert "Business Impact" in formatted_answer.content or "business_impact" in formatted_answer.content.lower()
    
    def test_format_analytics_answer(self):
        """Test formatting for analytics domain"""
        context = FormattingContext(
            domain=DomainType.ANALYTICS,
            output_format=OutputFormat.STRUCTURED_TEXT,
            style=FormattingStyle.TECHNICAL,
            target_audience="data_analysts"
        )
        
        formatted_answer = self.formatter.format_answer(
            holistic_answer=self.sample_holistic_answer,
            domain_knowledge=self.sample_domain_knowledge,
            intent_analysis=self.sample_intent_analysis,
            formatting_context=context
        )
        
        assert formatted_answer is not None
        assert formatted_answer.domain == DomainType.ANALYTICS
        assert len(formatted_answer.sections) > 0
        
        # Check for analytics-specific sections
        section_titles = [section['title'] for section in formatted_answer.sections]
        assert any("Data" in title or "Analysis" in title or "Insights" in title for title in section_titles)
    
    def test_markdown_output_format(self):
        """Test markdown output formatting"""
        context = FormattingContext(
            domain=DomainType.TECHNICAL,
            output_format=OutputFormat.MARKDOWN,
            style=FormattingStyle.TECHNICAL,
            target_audience="technical_team"
        )
        
        formatted_answer = self.formatter.format_answer(
            holistic_answer=self.sample_holistic_answer,
            domain_knowledge=self.sample_domain_knowledge,
            intent_analysis=self.sample_intent_analysis,
            formatting_context=context
        )
        
        assert formatted_answer is not None
        assert formatted_answer.format_type == OutputFormat.MARKDOWN
        # Check for markdown formatting
        assert "#" in formatted_answer.content  # Headers
        assert "##" in formatted_answer.content  # Subheaders
    
    def test_terminology_mapping(self):
        """Test domain-specific terminology mapping"""
        from datetime import datetime
        
        # Create a holistic answer with generic terms
        generic_answer = HolisticAnswer(
            answer_id="test_terminology_001",
            query_summary="What is the problem with our system?",
            executive_summary="The main issue is poor performance. We need a solution to improve the system.",
            main_sections=[
                AnswerSection(
                    title="Problem Analysis",
                    content="The issue appears to be in the data processing problem.",
                    priority=1,
                    section_type="analysis",
                    confidence=0.8
                )
            ],
            key_insights=["The problem is performance-related"],
            recommendations=["Implement a solution for the issue"],
            next_steps=["Address the problem systematically"],
            confidence_score=0.8,
            quality_metrics={"relevance": 0.8},
            synthesis_metadata={"test_mode": True},
            generated_at=datetime.now(),
            synthesis_time=1.0
        )
        
        # Test technical domain terminology
        technical_context = FormattingContext(
            domain=DomainType.TECHNICAL,
            output_format=OutputFormat.STRUCTURED_TEXT,
            style=FormattingStyle.TECHNICAL,
            target_audience="technical_team"
        )
        
        formatted_answer = self.formatter.format_answer(
            holistic_answer=generic_answer,
            domain_knowledge=self.sample_domain_knowledge,
            intent_analysis=self.sample_intent_analysis,
            formatting_context=technical_context
        )
        
        assert formatted_answer is not None
        # Check that terminology has been mapped (technical domain maps "issue" to "technical challenge")
        assert "technical challenge" in formatted_answer.content or "system issue" in formatted_answer.content
    
    def test_visualization_generation(self):
        """Test visualization generation for different domains"""
        context = FormattingContext(
            domain=DomainType.ANALYTICS,
            output_format=OutputFormat.DASHBOARD,
            style=FormattingStyle.TECHNICAL,
            target_audience="data_analysts",
            include_visualizations=True
        )
        
        formatted_answer = self.formatter.format_answer(
            holistic_answer=self.sample_holistic_answer,
            domain_knowledge=self.sample_domain_knowledge,
            intent_analysis=self.sample_intent_analysis,
            formatting_context=context
        )
        
        assert formatted_answer is not None
        assert len(formatted_answer.visualizations) > 0
        
        # Check visualization structure
        for viz in formatted_answer.visualizations:
            assert "type" in viz
            assert "title" in viz
            assert "description" in viz
    
    def test_domain_rules_retrieval(self):
        """Test retrieval of domain-specific rules"""
        # Test technical domain rules
        technical_rules = self.formatter.get_domain_rules(DomainType.TECHNICAL)
        assert technical_rules is not None
        assert technical_rules.domain == DomainType.TECHNICAL
        assert "technical_analysis" in technical_rules.section_order
        assert "implementation" in technical_rules.section_order
        
        # Test business domain rules
        business_rules = self.formatter.get_domain_rules(DomainType.BUSINESS)
        assert business_rules is not None
        assert business_rules.domain == DomainType.BUSINESS
        assert "executive_summary" in business_rules.section_order
        assert "business_impact" in business_rules.section_order
    
    def test_formatting_context_validation(self):
        """Test formatting context validation"""
        # Valid context
        valid_context = FormattingContext(
            domain=DomainType.TECHNICAL,
            output_format=OutputFormat.STRUCTURED_TEXT,
            style=FormattingStyle.TECHNICAL,
            target_audience="technical_team"
        )
        
        assert self.formatter.validate_formatting_context(valid_context) is True
        
        # Test with invalid context (None target audience)
        invalid_context = FormattingContext(
            domain=DomainType.TECHNICAL,
            output_format=OutputFormat.STRUCTURED_TEXT,
            style=FormattingStyle.TECHNICAL,
            target_audience=None
        )
        
        assert self.formatter.validate_formatting_context(invalid_context) is False
    
    def test_multiple_output_formats(self):
        """Test multiple output formats for the same content"""
        formats_to_test = [
            OutputFormat.STRUCTURED_TEXT,
            OutputFormat.MARKDOWN,
            OutputFormat.EXECUTIVE_SUMMARY,
            OutputFormat.TECHNICAL_REPORT
        ]
        
        for output_format in formats_to_test:
            context = FormattingContext(
                domain=DomainType.TECHNICAL,
                output_format=output_format,
                style=FormattingStyle.TECHNICAL,
                target_audience="technical_team"
            )
            
            formatted_answer = self.formatter.format_answer(
                holistic_answer=self.sample_holistic_answer,
                domain_knowledge=self.sample_domain_knowledge,
                intent_analysis=self.sample_intent_analysis,
                formatting_context=context
            )
            
            assert formatted_answer is not None
            assert formatted_answer.format_type == output_format
            assert len(formatted_answer.content) > 0
    
    def test_domain_specific_sections(self):
        """Test that domain-specific sections are correctly generated"""
        # Test financial domain
        financial_context = FormattingContext(
            domain=DomainType.FINANCIAL,
            output_format=OutputFormat.STRUCTURED_TEXT,
            style=FormattingStyle.FORMAL,
            target_audience="financial_analysts"
        )
        
        formatted_answer = self.formatter.format_answer(
            holistic_answer=self.sample_holistic_answer,
            domain_knowledge=self.sample_domain_knowledge,
            intent_analysis=self.sample_intent_analysis,
            formatting_context=financial_context
        )
        
        assert formatted_answer is not None
        section_types = [section['type'] for section in formatted_answer.sections]
        
        # Financial domain should have financial analysis and risk assessment
        assert any("financial" in section_type or "risk" in section_type for section_type in section_types)
    
    def test_supported_domains_and_formats(self):
        """Test that all supported domains and formats are accessible"""
        supported_domains = self.formatter.get_supported_domains()
        supported_formats = self.formatter.get_supported_formats()
        
        assert len(supported_domains) > 0
        assert len(supported_formats) > 0
        
        # Check that all enum values are supported
        assert DomainType.TECHNICAL in supported_domains
        assert DomainType.BUSINESS in supported_domains
        assert DomainType.ANALYTICS in supported_domains
        
        assert OutputFormat.STRUCTURED_TEXT in supported_formats
        assert OutputFormat.MARKDOWN in supported_formats
        assert OutputFormat.EXECUTIVE_SUMMARY in supported_formats
    
    def test_error_handling(self):
        """Test error handling in formatting process"""
        # Test with None inputs
        with pytest.raises(Exception):
            self.formatter.format_answer(
                holistic_answer=None,
                domain_knowledge=self.sample_domain_knowledge,
                intent_analysis=self.sample_intent_analysis,
                formatting_context=FormattingContext(
                    domain=DomainType.TECHNICAL,
                    output_format=OutputFormat.STRUCTURED_TEXT,
                    style=FormattingStyle.TECHNICAL,
                    target_audience="technical_team"
                )
            )
    
    def test_quality_metrics_preservation(self):
        """Test that quality metrics are preserved in formatted output"""
        context = FormattingContext(
            domain=DomainType.TECHNICAL,
            output_format=OutputFormat.STRUCTURED_TEXT,
            style=FormattingStyle.TECHNICAL,
            target_audience="technical_team"
        )
        
        formatted_answer = self.formatter.format_answer(
            holistic_answer=self.sample_holistic_answer,
            domain_knowledge=self.sample_domain_knowledge,
            intent_analysis=self.sample_intent_analysis,
            formatting_context=context
        )
        
        assert formatted_answer is not None
        assert "quality_metrics" in formatted_answer.metadata
        assert "confidence_score" in formatted_answer.metadata
        assert formatted_answer.metadata["confidence_score"] == self.sample_holistic_answer.confidence_score


if __name__ == "__main__":
    pytest.main([__file__]) 