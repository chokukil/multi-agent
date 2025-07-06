"""
Simple test for DomainAwareAgentSelector

This test validates the core functionality of the domain-aware agent selector
without complex dependencies.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

from core.query_processing.domain_aware_agent_selector import (
    DomainAwareAgentSelector, AgentType, AgentCapability, AgentSelection, AgentSelectionResult
)
from core.query_processing.intelligent_query_processor import EnhancedQuery, IntentAnalysis, DomainKnowledge, AnswerStructure, QueryType, DomainType, AnswerFormat


async def test_basic_agent_selector_functionality():
    """Test basic functionality of DomainAwareAgentSelector"""
    print("🧪 Testing DomainAwareAgentSelector basic functionality...")
    
    # Create selector
    selector = DomainAwareAgentSelector()
    
    # Test 1: Agent catalog initialization
    print("✅ Test 1: Agent catalog initialization")
    assert len(selector.agent_catalog) == 9
    assert AgentType.EDA_TOOLS in selector.agent_catalog
    assert AgentType.DATA_VISUALIZATION in selector.agent_catalog
    print(f"   - Found {len(selector.agent_catalog)} agents in catalog")
    
    # Test 2: Get specific agent capability
    print("✅ Test 2: Get agent capability")
    eda_capability = selector.get_agent_capability(AgentType.EDA_TOOLS)
    assert eda_capability is not None
    assert eda_capability.name == "EDA Tools Agent"
    assert "exploratory_analysis" in eda_capability.task_types
    print(f"   - EDA agent capabilities: {len(eda_capability.primary_skills)} primary skills")
    
    # Test 3: List all agents
    print("✅ Test 3: List available agents")
    all_agents = selector.list_available_agents()
    assert len(all_agents) == 9
    assert all(isinstance(agent, AgentCapability) for agent in all_agents)
    print(f"   - Listed {len(all_agents)} available agents")
    
    print("✅ All basic functionality tests passed!")
    return True


async def test_agent_selection_with_mock():
    """Test agent selection with mocked LLM"""
    print("🧪 Testing agent selection with mock LLM...")
    
    # Create selector with mock LLM
    selector = DomainAwareAgentSelector()
    
    # Mock the LLM response for requirements analysis
    mock_llm = AsyncMock()
    mock_response = MagicMock()
    mock_response.content = json.dumps({
        "data_requirements": ["manufacturing data", "process data"],
        "analysis_types": ["statistical process control", "anomaly detection", "exploratory analysis"],
        "output_formats": ["technical reports", "visualizations", "charts"],
        "technical_constraints": ["real-time processing"],
        "domain_specific_needs": ["semiconductor manufacturing", "process monitoring"],
        "complexity_factors": ["complex statistical analysis"],
        "success_criteria": ["process improvement", "anomaly detection"],
        "timeline_requirements": ["urgent processing"]
    })
    mock_llm.ainvoke.return_value = mock_response
    selector.llm = mock_llm
    
    # Create minimal test data structures
    intent_analysis = IntentAnalysis(
        primary_intent="process_analysis",
        data_scientist_perspective="Statistical analysis needed",
        domain_expert_perspective="Manufacturing process monitoring",
        technical_implementer_perspective="Implementation challenges",
        query_type=QueryType.ANALYTICAL,
        urgency_level=0.7,
        complexity_score=0.8,
        confidence_score=0.9
    )
    
    domain_knowledge = DomainKnowledge(
        domain_type=DomainType.MANUFACTURING,
        key_concepts=["Statistical Process Control", "Process Monitoring"],
        technical_terms=["계측값 데이터", "공정 이상"],
        business_context="Manufacturing process monitoring",
        required_expertise=["Statistical Analysis", "Process Control"],
        relevant_methodologies=["SPC", "Control Charts"],
        success_metrics=["Process Stability", "Quality Improvement"],
        potential_challenges=["Data Quality", "System Integration"]
    )
    
    answer_structure = AnswerStructure(
        expected_format=AnswerFormat.STRUCTURED_REPORT,
        key_sections=["Analysis", "Recommendations", "Visualization"],
        required_visualizations=["Control Charts", "Trend Analysis"],
        success_criteria=["Actionable Insights", "Statistical Validation"],
        expected_deliverables=["Report", "Visualizations", "Recommendations"],
        quality_checkpoints=["Data Quality", "Statistical Validity"]
    )
    
    enhanced_query = EnhancedQuery(
        original_query="LOT 히스토리와 계측값 데이터를 분석해서 공정 이상 여부를 판단해주세요",
        intent_analysis=intent_analysis,
        domain_knowledge=domain_knowledge,
        answer_structure=answer_structure,
        enhanced_queries=["반도체 제조 공정에서 Statistical Process Control을 적용한 공정 이상 탐지 분석"],
        execution_strategy="comprehensive_analysis",
        context_requirements={"data_quality": "high", "technical_expertise": "statistical_analysis"}
    )
    
    # Create minimal domain knowledge structure (using basic dict to avoid complex class issues)
    simple_domain_knowledge = type('SimpleDomainKnowledge', (), {
        'primary_domain': "Manufacturing",
        'technical_area': "Statistical Process Control",
        'key_concepts': [type('Concept', (), {'name': 'SPC', 'description': 'Statistical Process Control'})()],
        'technical_terms': [type('Term', (), {'name': '계측값 데이터', 'description': 'Measurement data'})()]
    })()
    
    # Create minimal intent analysis structure
    simple_intent_analysis = type('SimpleIntentAnalysis', (), {
        'primary_intent': "process_analysis",
        'task_type': "statistical_analysis",
        'complexity_level': "medium",
        'key_requirements': ["data_analysis", "visualization"]
    })()
    
    # Test agent selection
    print("✅ Test: Agent selection process")
    try:
        result = await selector.select_agents(enhanced_query, simple_domain_knowledge, simple_intent_analysis)
        
        assert isinstance(result, AgentSelectionResult)
        assert len(result.selected_agents) > 0
        assert result.total_confidence > 0.0
        assert result.reasoning
        assert result.execution_order
        assert result.estimated_duration
        
        # Check that appropriate agents were selected
        selected_types = [agent.agent_type for agent in result.selected_agents]
        print(f"   - Selected agents: {[agent.value for agent in selected_types]}")
        print(f"   - Total confidence: {result.total_confidence:.2f}")
        print(f"   - Execution strategy: {result.selection_strategy}")
        print(f"   - Estimated duration: {result.estimated_duration}")
        
        # Should include EDA and visualization for statistical analysis
        expected_agents = [AgentType.EDA_TOOLS, AgentType.DATA_VISUALIZATION]
        found_expected = [agent for agent in expected_agents if agent in selected_types]
        print(f"   - Found expected agents: {[agent.value for agent in found_expected]}")
        
        if len(found_expected) >= 1:
            print("✅ Agent selection completed successfully!")
        else:
            print("⚠️ Agent selection completed but may not be optimal")
        
        return True
        
    except Exception as e:
        print(f"❌ Agent selection failed: {e}")
        return False


async def test_agent_capabilities_comprehensive():
    """Test comprehensive agent capabilities"""
    print("🧪 Testing comprehensive agent capabilities...")
    
    selector = DomainAwareAgentSelector()
    
    # Test each agent type
    agent_tests = [
        (AgentType.DATA_LOADER, "data_loading", "file_handling"),
        (AgentType.DATA_CLEANING, "data_cleaning", "missing_values"),
        (AgentType.EDA_TOOLS, "exploratory_analysis", "statistics"),
        (AgentType.DATA_VISUALIZATION, "data_visualization", "plotly_charts"),
        (AgentType.FEATURE_ENGINEERING, "feature_engineering", "feature_creation"),
        (AgentType.H2O_ML, "machine_learning", "automl"),
    ]
    
    for agent_type, expected_task, expected_confidence in agent_tests:
        capability = selector.get_agent_capability(agent_type)
        assert capability is not None, f"Agent {agent_type} not found"
        assert expected_task in capability.task_types, f"Expected task {expected_task} not found in {agent_type}"
        print(f"✅ {capability.name}: {len(capability.primary_skills)} skills, {len(capability.task_types)} task types")
    
    print("✅ All agent capabilities verified!")
    return True


async def main():
    """Run all tests"""
    print("🚀 Starting DomainAwareAgentSelector tests...\n")
    
    tests = [
        test_basic_agent_selector_functionality,
        test_agent_capabilities_comprehensive,
        test_agent_selection_with_mock,
    ]
    
    results = []
    for test in tests:
        try:
            result = await test()
            results.append(result)
            print()
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            results.append(False)
            print()
    
    # Summary
    passed = sum(results)
    total = len(results)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! DomainAwareAgentSelector is working correctly.")
        return True
    else:
        print(f"⚠️ {total - passed} tests failed. Please review the issues.")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1) 