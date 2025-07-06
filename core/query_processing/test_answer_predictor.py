"""
Test script for AnswerStructurePredictor
"""

import asyncio
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from core.query_processing.answer_predictor import AnswerStructurePredictor
from core.query_processing.intent_analyzer import (
    MultiPerspectiveIntentAnalyzer, 
    DetailedIntentAnalysis, 
    QueryComplexity, 
    UrgencyLevel
)
from core.query_processing.domain_extractor import DomainKnowledgeExtractor
from core.query_processing.intelligent_query_processor import QueryType

async def test_answer_predictor():
    """Test the AnswerStructurePredictor with comprehensive inputs"""
    
    print("🚀 Testing AnswerStructurePredictor...")
    
    # Initialize components
    intent_analyzer = MultiPerspectiveIntentAnalyzer()
    domain_extractor = DomainKnowledgeExtractor()
    answer_predictor = AnswerStructurePredictor()
    
    # Test query (complex semiconductor manufacturing scenario)
    test_query = """
    LOT 히스토리와 계측값 데이터를 분석해서 공정 이상 여부를 판단하고, 
    이상 원인을 분석해서 기술적 조치 방향을 제안해주세요. 
    또한 향후 유사 문제 예방을 위한 모니터링 시스템도 제안하고,
    품질 관리 체계와 연계하여 전사적 품질 개선 방안을 도출해주세요.
    반도체 공정 특성을 고려한 통계적 품질 관리 방법론도 함께 제시해주세요.
    """
    
    # Sample data context
    data_context = {
        "available_datasets": [
            "ion_implant_3lot_dataset.xlsx",
            "lot_history_data.csv",
            "measurement_data.csv",
            "equipment_logs.csv",
            "quality_control_records.csv"
        ],
        "stakeholders": ["production_manager", "process_engineer", "quality_assurance", "plant_manager"],
        "urgency": "high",
        "business_impact": "high"
    }
    
    try:
        print(f"📋 Running complete pipeline for answer structure prediction...")
        
        # Step 1: Analyze intent (Multi-perspective)
        print("🔍 Step 1: Multi-perspective intent analysis...")
        intent_analysis = await intent_analyzer.analyze_intent_comprehensive(test_query, data_context)
        
        # Step 2: Extract domain knowledge
        print("🧠 Step 2: Domain knowledge extraction...")
        domain_knowledge = await domain_extractor.extract_comprehensive_domain_knowledge(
            test_query, 
            {
                "primary_intent": intent_analysis.primary_intent,
                "query_type": intent_analysis.query_type.value,
                "complexity_level": intent_analysis.complexity_level.value
            }, 
            data_context
        )
        
        # Step 3: Predict answer structure
        print("🔮 Step 3: Answer structure prediction...")
        predicted_structure = await answer_predictor.predict_optimal_structure(
            intent_analysis, 
            domain_knowledge, 
            data_context
        )
        
        # Display comprehensive results
        print("\n" + "="*80)
        print("🎯 COMPREHENSIVE ANSWER STRUCTURE PREDICTION RESULTS")
        print("="*80)
        
        # Primary Template Details
        template = predicted_structure.primary_template
        print(f"\n📋 Primary Answer Template:")
        print(f"   • Format Type: {template.format_type.value}")
        print(f"   • Target Audience: {template.target_audience}")
        print(f"   • Complexity Score: {template.complexity_score:.2f}")
        print(f"   • Estimated Completion Time: {template.estimated_completion_time}")
        
        # Sections
        print(f"\n📚 Answer Sections ({len(template.sections)}):")
        for i, section in enumerate(template.sections, 1):
            print(f"   {i}. {section.title} [{section.section_type.value}]")
            print(f"      • Priority: {section.priority}/10")
            print(f"      • Length: {section.estimated_length}")
            print(f"      • Description: {section.description}")
        
        # Visualizations
        print(f"\n📊 Required Visualizations ({len(template.required_visualizations)}):")
        for viz in template.required_visualizations:
            print(f"   • {viz.value.replace('_', ' ').title()}")
        
        # Quality Checkpoints
        print(f"\n✅ Quality Checkpoints ({len(template.quality_checkpoints)}):")
        for checkpoint in template.quality_checkpoints:
            print(f"   • {checkpoint.value.replace('_', ' ').title()}")
        
        # Alternative Templates
        print(f"\n🔄 Alternative Templates ({len(predicted_structure.alternative_templates)}):")
        for i, alt_template in enumerate(predicted_structure.alternative_templates, 1):
            print(f"   {i}. {alt_template.format_type.value} (Target: {alt_template.target_audience})")
            print(f"      • Complexity: {alt_template.complexity_score:.2f}")
            print(f"      • Time: {alt_template.estimated_completion_time}")
            print(f"      • Sections: {len(alt_template.sections)}")
        
        # Customizations
        print(f"\n🎛️ Answer Customizations:")
        if predicted_structure.customizations:
            for key, value in predicted_structure.customizations.items():
                print(f"   • {key.title()}: {value}")
        else:
            print("   • No specific customizations")
        
        # Adaptation Reasoning
        print(f"\n💡 Adaptation Reasoning:")
        print(f"   {predicted_structure.adaptation_reasoning}")
        
        # Validation Criteria
        print(f"\n🔍 Validation Criteria:")
        for criterion in predicted_structure.validation_criteria:
            print(f"   • {criterion}")
        
        print(f"\n✅ Overall Prediction Confidence: {predicted_structure.confidence_score:.2f}")
        
        print(f"\n" + "="*50)
        print("📝 STRUCTURE PREDICTION SUMMARY")
        print("="*50)
        print(answer_predictor.get_structure_summary(predicted_structure))
        
        # Show integration summary
        print(f"\n" + "="*50)
        print("🎯 INTEGRATION SUMMARY")
        print("="*50)
        print(f"✅ Intent Analysis Confidence: {intent_analysis.overall_confidence:.2f}")
        print(f"✅ Domain Knowledge Confidence: {domain_knowledge.extraction_confidence:.2f}")
        print(f"✅ Structure Prediction Confidence: {predicted_structure.confidence_score:.2f}")
        print(f"📊 Overall Pipeline Confidence: {(intent_analysis.overall_confidence + domain_knowledge.extraction_confidence + predicted_structure.confidence_score) / 3:.2f}")
        
        print("\n✅ Answer structure prediction test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_answer_predictor()) 