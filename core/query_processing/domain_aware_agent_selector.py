"""
Domain-Aware Agent Selector for CherryAI

This module intelligently selects appropriate A2A agents based on:
1. Domain knowledge analysis from Phase 1
2. Intent analysis results
3. Task requirements and capabilities
4. Agent expertise mapping
"""

import asyncio
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from core.llm_factory import create_llm_instance
from core.query_processing.intelligent_query_processor import EnhancedQuery
from core.query_processing.domain_extractor import DomainKnowledge
from core.query_processing.intent_analyzer import IntentAnalysis

logger = logging.getLogger(__name__)


class AgentType(Enum):
    """Available A2A Agent Types"""
    DATA_LOADER = "data_loader"
    DATA_CLEANING = "data_cleaning"
    DATA_WRANGLING = "data_wrangling"
    EDA_TOOLS = "eda_tools"
    DATA_VISUALIZATION = "data_visualization"
    FEATURE_ENGINEERING = "feature_engineering"
    H2O_ML = "h2o_ml"
    MLFLOW_TOOLS = "mlflow_tools"
    SQL_DATABASE = "sql_database"


@dataclass
class AgentCapability:
    """Agent capability definition"""
    agent_type: AgentType
    name: str
    description: str
    primary_skills: List[str]
    secondary_skills: List[str]
    domain_expertise: List[str]
    task_types: List[str]
    prerequisites: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    port: int = 8300
    confidence_factors: Dict[str, float] = field(default_factory=dict)


@dataclass
class AgentSelection:
    """Selected agent with reasoning"""
    agent_type: AgentType
    confidence: float
    reasoning: str
    priority: int
    dependencies: List[AgentType] = field(default_factory=list)
    expected_outputs: List[str] = field(default_factory=list)
    domain_relevance: float = 0.0
    task_fit: float = 0.0


@dataclass
class AgentSelectionResult:
    """Complete agent selection result"""
    selected_agents: List[AgentSelection]
    selection_strategy: str
    total_confidence: float
    reasoning: str
    execution_order: List[AgentType]
    estimated_duration: str
    success_probability: float
    alternative_options: List[Dict[str, Any]] = field(default_factory=list)


class DomainAwareAgentSelector:
    """
    Intelligent agent selector that chooses appropriate A2A agents
    based on domain knowledge and task analysis
    """
    
    def __init__(self):
        self.llm = create_llm_instance()
        self.agent_catalog = self._initialize_agent_catalog()
        logger.info("DomainAwareAgentSelector initialized")
    
    def _initialize_agent_catalog(self) -> Dict[AgentType, AgentCapability]:
        """Initialize comprehensive agent capability catalog"""
        catalog = {}
        
        # Data Loader Agent
        catalog[AgentType.DATA_LOADER] = AgentCapability(
            agent_type=AgentType.DATA_LOADER,
            name="Data Loader Agent",
            description="Specialized in loading and importing data from various sources",
            primary_skills=[
                "File format handling (CSV, Excel, JSON, Parquet)",
                "Database connection and queries",
                "Data source integration",
                "Data validation and schema detection"
            ],
            secondary_skills=[
                "Data format conversion",
                "Basic data profiling",
                "Error handling and recovery"
            ],
            domain_expertise=[
                "Data ingestion pipelines",
                "ETL processes",
                "Data warehouse integration",
                "Cloud storage access"
            ],
            task_types=[
                "data_loading", "file_import", "database_query",
                "data_ingestion", "source_integration"
            ],
            outputs=["loaded_dataframes", "data_schemas", "loading_reports"],
            port=8301,
            confidence_factors={
                "file_handling": 0.9,
                "database_operations": 0.8,
                "data_validation": 0.7
            }
        )
        
        # Data Cleaning Agent
        catalog[AgentType.DATA_CLEANING] = AgentCapability(
            agent_type=AgentType.DATA_CLEANING,
            name="Data Cleaning Agent",
            description="Expert in data quality improvement and preprocessing",
            primary_skills=[
                "Missing value handling",
                "Outlier detection and treatment",
                "Data type conversion",
                "Duplicate removal",
                "Data standardization"
            ],
            secondary_skills=[
                "Data profiling",
                "Quality metrics calculation",
                "Data validation rules"
            ],
            domain_expertise=[
                "Data quality management",
                "Statistical cleaning methods",
                "Business rule validation",
                "Data governance compliance"
            ],
            task_types=[
                "data_cleaning", "preprocessing", "quality_improvement",
                "outlier_detection", "missing_value_handling"
            ],
            prerequisites=["loaded_data"],
            outputs=["cleaned_datasets", "quality_reports", "cleaning_logs"],
            port=8302,
            confidence_factors={
                "missing_values": 0.9,
                "outliers": 0.8,
                "data_types": 0.85
            }
        )
        
        # Data Wrangling Agent
        catalog[AgentType.DATA_WRANGLING] = AgentCapability(
            agent_type=AgentType.DATA_WRANGLING,
            name="Data Wrangling Agent",
            description="Specialized in data transformation and manipulation",
            primary_skills=[
                "Data reshaping (pivot, melt, stack)",
                "Column operations and transformations",
                "Aggregation and grouping",
                "Data merging and joining",
                "Time series manipulation"
            ],
            secondary_skills=[
                "Custom transformations",
                "Data validation",
                "Performance optimization"
            ],
            domain_expertise=[
                "Data pipeline engineering",
                "Business logic implementation",
                "Data architecture patterns",
                "Performance tuning"
            ],
            task_types=[
                "data_transformation", "reshaping", "aggregation",
                "merging", "time_series_processing"
            ],
            prerequisites=["cleaned_data"],
            outputs=["transformed_datasets", "transformation_logs"],
            port=8303,
            confidence_factors={
                "transformations": 0.9,
                "aggregations": 0.85,
                "merging": 0.8
            }
        )
        
        # EDA Tools Agent
        catalog[AgentType.EDA_TOOLS] = AgentCapability(
            agent_type=AgentType.EDA_TOOLS,
            name="EDA Tools Agent",
            description="Expert in exploratory data analysis and statistical investigation",
            primary_skills=[
                "Statistical analysis and summaries",
                "Distribution analysis",
                "Correlation analysis",
                "Hypothesis testing",
                "Data profiling and characterization"
            ],
            secondary_skills=[
                "Automated insights generation",
                "Anomaly detection",
                "Pattern recognition"
            ],
            domain_expertise=[
                "Statistical methods",
                "Data science methodology",
                "Business intelligence",
                "Research analytics"
            ],
            task_types=[
                "exploratory_analysis", "statistical_analysis", "profiling",
                "correlation_analysis", "hypothesis_testing", "anomaly_detection"
            ],
            prerequisites=["processed_data"],
            outputs=["statistical_reports", "analysis_insights", "profiling_results"],
            port=8312,
            confidence_factors={
                "statistics": 0.9,
                "correlations": 0.85,
                "distributions": 0.8
            }
        )
        
        # Data Visualization Agent
        catalog[AgentType.DATA_VISUALIZATION] = AgentCapability(
            agent_type=AgentType.DATA_VISUALIZATION,
            name="Data Visualization Agent",
            description="Specialized in creating interactive and static visualizations",
            primary_skills=[
                "Interactive charts (Plotly, Bokeh)",
                "Statistical plots (matplotlib, seaborn)",
                "Dashboard creation",
                "Custom visualization design",
                "Chart optimization and styling"
            ],
            secondary_skills=[
                "Color theory and design",
                "UX/UI for data",
                "Storytelling with data"
            ],
            domain_expertise=[
                "Data visualization best practices",
                "Business reporting",
                "Scientific visualization",
                "Executive dashboards"
            ],
            task_types=[
                "data_visualization", "charting", "dashboard_creation",
                "plotting", "visual_analysis", "reporting"
            ],
            prerequisites=["analyzed_data"],
            outputs=["interactive_charts", "static_plots", "dashboards", "visual_reports"],
            port=8308,
            confidence_factors={
                "plotly_charts": 0.9,
                "statistical_plots": 0.85,
                "dashboards": 0.8
            }
        )
        
        # Feature Engineering Agent
        catalog[AgentType.FEATURE_ENGINEERING] = AgentCapability(
            agent_type=AgentType.FEATURE_ENGINEERING,
            name="Feature Engineering Agent",
            description="Expert in creating and optimizing features for machine learning",
            primary_skills=[
                "Feature creation and selection",
                "Encoding categorical variables",
                "Scaling and normalization",
                "Dimensionality reduction",
                "Time-based feature engineering"
            ],
            secondary_skills=[
                "Feature importance analysis",
                "Automated feature generation",
                "Feature validation"
            ],
            domain_expertise=[
                "Machine learning preprocessing",
                "Statistical feature engineering",
                "Domain-specific feature creation",
                "Feature optimization"
            ],
            task_types=[
                "feature_engineering", "feature_selection", "encoding",
                "scaling", "dimensionality_reduction", "ml_preprocessing"
            ],
            prerequisites=["cleaned_data"],
            outputs=["engineered_features", "feature_reports", "ml_ready_datasets"],
            port=8309,
            confidence_factors={
                "feature_creation": 0.9,
                "encoding": 0.85,
                "scaling": 0.8
            }
        )
        
        # H2O ML Agent
        catalog[AgentType.H2O_ML] = AgentCapability(
            agent_type=AgentType.H2O_ML,
            name="H2O ML Agent",
            description="Specialized in machine learning model development with H2O",
            primary_skills=[
                "AutoML model training",
                "Model evaluation and selection",
                "Hyperparameter tuning",
                "Model interpretation",
                "Performance metrics analysis"
            ],
            secondary_skills=[
                "Model deployment preparation",
                "Feature importance analysis",
                "Model comparison"
            ],
            domain_expertise=[
                "Machine learning algorithms",
                "AutoML best practices",
                "Model evaluation methodologies",
                "Production ML systems"
            ],
            task_types=[
                "machine_learning", "model_training", "automl",
                "prediction", "classification", "regression"
            ],
            prerequisites=["engineered_features"],
            outputs=["trained_models", "model_reports", "predictions", "performance_metrics"],
            port=8313,
            confidence_factors={
                "automl": 0.9,
                "model_evaluation": 0.85,
                "predictions": 0.8
            }
        )
        
        # MLflow Tools Agent
        catalog[AgentType.MLFLOW_TOOLS] = AgentCapability(
            agent_type=AgentType.MLFLOW_TOOLS,
            name="MLflow Tools Agent",
            description="Expert in ML model tracking and lifecycle management",
            primary_skills=[
                "Model versioning and tracking",
                "Experiment management",
                "Model registry operations",
                "Performance monitoring",
                "Deployment coordination"
            ],
            secondary_skills=[
                "Model comparison",
                "Artifact management",
                "Model serving"
            ],
            domain_expertise=[
                "ML operations (MLOps)",
                "Model lifecycle management",
                "Experiment tracking",
                "Model governance"
            ],
            task_types=[
                "model_tracking", "experiment_management", "mlops",
                "model_registry", "version_control", "deployment"
            ],
            prerequisites=["trained_models"],
            outputs=["tracking_reports", "model_registry", "deployment_configs"],
            port=8314,
            confidence_factors={
                "tracking": 0.9,
                "registry": 0.85,
                "deployment": 0.8
            }
        )
        
        # SQL Database Agent
        catalog[AgentType.SQL_DATABASE] = AgentCapability(
            agent_type=AgentType.SQL_DATABASE,
            name="SQL Database Agent",
            description="Specialized in database operations and SQL-based analysis",
            primary_skills=[
                "Complex SQL query generation",
                "Database schema analysis",
                "Performance optimization",
                "Data aggregation and reporting",
                "Database administration"
            ],
            secondary_skills=[
                "Query optimization",
                "Database design",
                "Data migration"
            ],
            domain_expertise=[
                "Database systems",
                "SQL optimization",
                "Data warehousing",
                "Business intelligence"
            ],
            task_types=[
                "database_analysis", "sql_queries", "data_extraction",
                "reporting", "database_operations"
            ],
            prerequisites=["database_connection"],
            outputs=["query_results", "database_reports", "optimized_queries"],
            port=8310,
            confidence_factors={
                "sql_generation": 0.9,
                "optimization": 0.8,
                "reporting": 0.85
            }
        )
        
        return catalog
    
    async def select_agents(
        self,
        enhanced_query: EnhancedQuery,
        domain_knowledge: DomainKnowledge,
        intent_analysis: IntentAnalysis
    ) -> AgentSelectionResult:
        """
        Select appropriate agents based on comprehensive analysis
        
        Args:
            enhanced_query: Enhanced query from Phase 1
            domain_knowledge: Domain knowledge extraction results
            intent_analysis: Intent analysis results
            
        Returns:
            AgentSelectionResult with selected agents and reasoning
        """
        
        logger.info("Starting intelligent agent selection process")
        
        try:
            # Analyze requirements
            requirements = await self._analyze_requirements(
                enhanced_query, domain_knowledge, intent_analysis
            )
            
            # Generate agent candidates
            candidates = await self._generate_agent_candidates(requirements)
            
            # Score and rank agents
            scored_agents = await self._score_and_rank_agents(
                candidates, requirements, domain_knowledge, intent_analysis
            )
            
            # Determine execution strategy
            execution_plan = await self._determine_execution_strategy(
                scored_agents, requirements
            )
            
            # Build final selection
            selection_result = await self._build_selection_result(
                execution_plan, requirements, scored_agents
            )
            
            logger.info(f"Agent selection completed: {len(selection_result.selected_agents)} agents selected")
            
            return selection_result
            
        except Exception as e:
            logger.error(f"Error in agent selection: {e}")
            # Return fallback selection
            return await self._create_fallback_selection(enhanced_query)
    
    async def _analyze_requirements(
        self,
        enhanced_query: EnhancedQuery,
        domain_knowledge: DomainKnowledge,
        intent_analysis: IntentAnalysis
    ) -> Dict[str, Any]:
        """Analyze requirements for agent selection"""
        
        prompt = f"""
        Analyze the following query and context to extract specific requirements for AI agent selection:

        **Enhanced Query**: {enhanced_query.enhanced_query}
        
        **Domain Knowledge**:
        - Primary Domain: {domain_knowledge.primary_domain}
        - Technical Area: {domain_knowledge.technical_area}
        - Key Concepts: {[concept.name for concept in domain_knowledge.key_concepts]}
        
        **Intent Analysis**:
        - Primary Intent: {intent_analysis.primary_intent}
        - Task Type: {intent_analysis.task_type}
        - Complexity Level: {intent_analysis.complexity_level}
        
        Extract and return the following requirements in JSON format:
        {{
            "data_requirements": ["required data types and sources"],
            "analysis_types": ["specific analysis methods needed"],
            "output_formats": ["expected output types"],
            "technical_constraints": ["technical limitations or requirements"],
            "domain_specific_needs": ["domain-specific requirements"],
            "complexity_factors": ["factors affecting complexity"],
            "success_criteria": ["criteria for success"],
            "timeline_requirements": ["timing and urgency factors"]
        }}
        """
        
        try:
            response = await self.llm.ainvoke(prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Parse JSON response
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                requirements = json.loads(json_match.group())
                logger.info("Requirements analysis completed successfully")
                return requirements
            else:
                logger.warning("Could not parse requirements JSON, using fallback")
                return self._create_fallback_requirements()
                
        except Exception as e:
            logger.error(f"Error in requirements analysis: {e}")
            return self._create_fallback_requirements()
    
    async def _generate_agent_candidates(self, requirements: Dict[str, Any]) -> List[AgentType]:
        """Generate candidate agents based on requirements"""
        
        candidates = []
        
        # Rule-based initial filtering
        analysis_types = requirements.get("analysis_types", [])
        output_formats = requirements.get("output_formats", [])
        data_requirements = requirements.get("data_requirements", [])
        
        # Check each agent type
        for agent_type, capability in self.agent_catalog.items():
            score = 0
            
            # Check task type alignment
            for task_type in capability.task_types:
                if any(task_type in analysis.lower() for analysis in analysis_types):
                    score += 2
            
            # Check output alignment
            for output in capability.outputs:
                if any(output in format.lower() for format in output_formats):
                    score += 1
            
            # Check skill alignment
            for skill in capability.primary_skills:
                if any(skill.lower() in req.lower() for req in data_requirements):
                    score += 1
            
            # Add to candidates if score is above threshold
            if score > 0:
                candidates.append(agent_type)
        
        logger.info(f"Generated {len(candidates)} agent candidates")
        return candidates
    
    async def _score_and_rank_agents(
        self,
        candidates: List[AgentType],
        requirements: Dict[str, Any],
        domain_knowledge: DomainKnowledge,
        intent_analysis: IntentAnalysis
    ) -> List[AgentSelection]:
        """Score and rank agent candidates"""
        
        scored_agents = []
        
        for agent_type in candidates:
            capability = self.agent_catalog[agent_type]
            
            # Calculate domain relevance
            domain_score = await self._calculate_domain_relevance(
                capability, domain_knowledge
            )
            
            # Calculate task fit
            task_score = await self._calculate_task_fit(
                capability, requirements, intent_analysis
            )
            
            # Calculate overall confidence
            confidence = (domain_score + task_score) / 2.0
            
            # Generate reasoning
            reasoning = await self._generate_selection_reasoning(
                capability, domain_score, task_score, requirements
            )
            
            selection = AgentSelection(
                agent_type=agent_type,
                confidence=confidence,
                reasoning=reasoning,
                priority=1,  # Will be updated in execution strategy
                domain_relevance=domain_score,
                task_fit=task_score,
                expected_outputs=capability.outputs
            )
            
            scored_agents.append(selection)
        
        # Sort by confidence
        scored_agents.sort(key=lambda x: x.confidence, reverse=True)
        
        logger.info(f"Scored and ranked {len(scored_agents)} agents")
        return scored_agents
    
    async def _calculate_domain_relevance(
        self, capability: AgentCapability, domain_knowledge: DomainKnowledge
    ) -> float:
        """Calculate domain relevance score"""
        
        score = 0.0
        max_score = 0.0
        
        # Check domain expertise alignment
        for expertise in capability.domain_expertise:
            max_score += 1.0
            if domain_knowledge.primary_domain.lower() in expertise.lower():
                score += 1.0
            elif domain_knowledge.technical_area.lower() in expertise.lower():
                score += 0.7
        
        # Check concept alignment
        for concept in domain_knowledge.key_concepts:
            max_score += 0.5
            if any(concept.name.lower() in skill.lower() for skill in capability.primary_skills):
                score += 0.5
        
        return score / max_score if max_score > 0 else 0.0
    
    async def _calculate_task_fit(
        self, capability: AgentCapability, requirements: Dict[str, Any], intent_analysis: IntentAnalysis
    ) -> float:
        """Calculate task fit score"""
        
        score = 0.0
        max_score = 0.0
        
        # Check task type alignment
        for task_type in capability.task_types:
            max_score += 1.0
            if task_type in intent_analysis.task_type.lower():
                score += 1.0
        
        # Check analysis types
        analysis_types = requirements.get("analysis_types", [])
        for analysis in analysis_types:
            max_score += 0.5
            if any(analysis.lower() in skill.lower() for skill in capability.primary_skills):
                score += 0.5
        
        return score / max_score if max_score > 0 else 0.0
    
    async def _generate_selection_reasoning(
        self, capability: AgentCapability, domain_score: float, task_score: float, requirements: Dict[str, Any]
    ) -> str:
        """Generate reasoning for agent selection"""
        
        reasoning = f"Selected {capability.name} with domain relevance {domain_score:.2f} and task fit {task_score:.2f}. "
        
        if domain_score > 0.7:
            reasoning += "Strong domain expertise match. "
        elif domain_score > 0.4:
            reasoning += "Moderate domain expertise match. "
        else:
            reasoning += "Limited domain expertise match. "
        
        if task_score > 0.7:
            reasoning += "Excellent task capability alignment. "
        elif task_score > 0.4:
            reasoning += "Good task capability alignment. "
        else:
            reasoning += "Basic task capability alignment. "
        
        # Add key capabilities
        top_skills = capability.primary_skills[:2]
        reasoning += f"Key capabilities: {', '.join(top_skills)}."
        
        return reasoning
    
    async def _determine_execution_strategy(
        self, scored_agents: List[AgentSelection], requirements: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Determine execution strategy and dependencies"""
        
        # Define standard execution order based on data pipeline
        standard_order = [
            AgentType.DATA_LOADER,
            AgentType.DATA_CLEANING,
            AgentType.DATA_WRANGLING,
            AgentType.EDA_TOOLS,
            AgentType.FEATURE_ENGINEERING,
            AgentType.DATA_VISUALIZATION,
            AgentType.H2O_ML,
            AgentType.MLFLOW_TOOLS,
            AgentType.SQL_DATABASE
        ]
        
        selected_types = [agent.agent_type for agent in scored_agents]
        
        # Create execution order based on standard pipeline
        execution_order = []
        for agent_type in standard_order:
            if agent_type in selected_types:
                execution_order.append(agent_type)
        
        # Add any remaining agents
        for agent_type in selected_types:
            if agent_type not in execution_order:
                execution_order.append(agent_type)
        
        # Assign priorities
        for i, agent in enumerate(scored_agents):
            if agent.agent_type in execution_order:
                agent.priority = execution_order.index(agent.agent_type) + 1
        
        return {
            "execution_order": execution_order,
            "strategy": "pipeline_based",
            "parallel_possible": False,  # Sequential execution for now
            "estimated_duration": self._estimate_duration(len(execution_order))
        }
    
    async def _build_selection_result(
        self, execution_plan: Dict[str, Any], requirements: Dict[str, Any], scored_agents: List[AgentSelection]
    ) -> AgentSelectionResult:
        """Build final selection result"""
        
        # Filter agents above confidence threshold
        confidence_threshold = 0.3
        selected_agents = [agent for agent in scored_agents if agent.confidence >= confidence_threshold]
        
        # If no agents meet threshold, select top 3
        if not selected_agents:
            selected_agents = scored_agents[:3]
        
        # Calculate overall confidence
        total_confidence = sum(agent.confidence for agent in selected_agents) / len(selected_agents)
        
        # Generate overall reasoning
        reasoning = f"Selected {len(selected_agents)} agents based on domain analysis and task requirements. "
        reasoning += f"Average confidence: {total_confidence:.2f}. "
        reasoning += f"Execution strategy: {execution_plan['strategy']}."
        
        return AgentSelectionResult(
            selected_agents=selected_agents,
            selection_strategy=execution_plan["strategy"],
            total_confidence=total_confidence,
            reasoning=reasoning,
            execution_order=execution_plan["execution_order"],
            estimated_duration=execution_plan["estimated_duration"],
            success_probability=min(total_confidence + 0.1, 1.0)
        )
    
    def _create_fallback_requirements(self) -> Dict[str, Any]:
        """Create fallback requirements"""
        return {
            "data_requirements": ["tabular data", "file input"],
            "analysis_types": ["exploratory analysis", "basic statistics"],
            "output_formats": ["reports", "visualizations"],
            "technical_constraints": ["standard processing"],
            "domain_specific_needs": ["general analysis"],
            "complexity_factors": ["medium complexity"],
            "success_criteria": ["actionable insights"],
            "timeline_requirements": ["standard processing time"]
        }
    
    async def _create_fallback_selection(self, enhanced_query: EnhancedQuery) -> AgentSelectionResult:
        """Create fallback selection when main process fails"""
        
        # Default agent selection for common scenarios
        default_agents = [
            AgentSelection(
                agent_type=AgentType.DATA_LOADER,
                confidence=0.8,
                reasoning="Default data loading capability",
                priority=1,
                expected_outputs=["loaded_dataframes"]
            ),
            AgentSelection(
                agent_type=AgentType.EDA_TOOLS,
                confidence=0.7,
                reasoning="Default exploratory analysis",
                priority=2,
                expected_outputs=["statistical_reports"]
            ),
            AgentSelection(
                agent_type=AgentType.DATA_VISUALIZATION,
                confidence=0.6,
                reasoning="Default visualization capability",
                priority=3,
                expected_outputs=["charts", "plots"]
            )
        ]
        
        return AgentSelectionResult(
            selected_agents=default_agents,
            selection_strategy="fallback_default",
            total_confidence=0.7,
            reasoning="Fallback selection due to processing error",
            execution_order=[AgentType.DATA_LOADER, AgentType.EDA_TOOLS, AgentType.DATA_VISUALIZATION],
            estimated_duration="10-15 minutes",
            success_probability=0.7
        )
    
    def _estimate_duration(self, num_agents: int) -> str:
        """Estimate execution duration based on number of agents"""
        base_time = 5  # minutes per agent
        total_minutes = num_agents * base_time
        
        if total_minutes < 10:
            return f"{total_minutes} minutes"
        elif total_minutes < 60:
            return f"{total_minutes} minutes"
        else:
            hours = total_minutes // 60
            minutes = total_minutes % 60
            return f"{hours}h {minutes}m"
    
    def get_agent_capability(self, agent_type: AgentType) -> Optional[AgentCapability]:
        """Get capability information for specific agent"""
        return self.agent_catalog.get(agent_type)
    
    def list_available_agents(self) -> List[AgentCapability]:
        """List all available agents"""
        return list(self.agent_catalog.values()) 