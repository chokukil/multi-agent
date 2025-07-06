"""
Test script for MultiPerspectiveIntentAnalyzer
"""

import asyncio
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from core.query_processing.intent_analyzer import MultiPerspectiveIntentAnalyzer

async def test_intent_analyzer():
    """Test the MultiPerspectiveIntentAnalyzer with a complex query"""
    
    print("🚀 Testing MultiPerspectiveIntentAnalyzer...")
    
    # Initialize the analyzer
    analyzer = MultiPerspectiveIntentAnalyzer()
    
    # Test query (complex manufacturing scenario)
    test_query = """
    LOT 히스토리와 계측값 데이터를 분석해서 공정 이상 여부를 판단하고, 
    이상 원인을 분석해서 기술적 조치 방향을 제안해주세요. 
    또한 향후 유사 문제 예방을 위한 모니터링 시스템도 제안해주세요.
    """
    
    # Sample data context
    data_context = {
        "available_datasets": [
            "ion_implant_3lot_dataset.xlsx",
            "lot_history_data.csv",
            "measurement_data.csv",
            "equipment_logs.csv"
        ],
        "data_columns": [
            "LOT_ID", "PROCESS_STEP", "MEASUREMENT_VALUE", 
            "SPEC_LIMIT", "TIMESTAMP", "EQUIPMENT_ID", "OPERATOR_ID"
        ],
        "domain": "semiconductor_manufacturing",
        "urgency": "high",
        "stakeholders": ["production_manager", "process_engineer", "quality_assurance"]
    }
    
    try:
        # Process the query with comprehensive analysis
        print(f"📋 Processing complex query: {test_query.strip()}")
        
        detailed_analysis = await analyzer.analyze_intent_comprehensive(test_query, data_context)
        
        # Display comprehensive results
        print("\n" + "="*80)
        print("🎯 MULTI-PERSPECTIVE INTENT ANALYSIS RESULTS")
        print("="*80)
        
        print(f"\n🎯 Primary Intent: {detailed_analysis.primary_intent}")
        print(f"📊 Secondary Intents: {', '.join(detailed_analysis.secondary_intents)}")
        print(f"🔍 Query Type: {detailed_analysis.query_type.value}")
        print(f"📈 Complexity: {detailed_analysis.complexity_level.value}")
        print(f"⚡ Urgency: {detailed_analysis.urgency_level.value}")
        print(f"🎯 Priority: {detailed_analysis.execution_priority}/10")
        print(f"⏱️ Timeline: {detailed_analysis.estimated_timeline}")
        print(f"🔗 Dependencies: {', '.join(detailed_analysis.critical_dependencies)}")
        print(f"✅ Overall Confidence: {detailed_analysis.overall_confidence:.2f}")
        
        print(f"\n" + "="*50)
        print("📊 PERSPECTIVE-BY-PERSPECTIVE ANALYSIS")
        print("="*50)
        
        for perspective_type, perspective in detailed_analysis.perspectives.items():
            print(f"\n🎭 {perspective_type.value.replace('_', ' ').title()} Perspective:")
            print(f"   • Primary Concerns: {', '.join(perspective.primary_concerns[:3])}")
            print(f"   • Methodology Suggestions: {', '.join(perspective.methodology_suggestions[:3])}")
            print(f"   • Potential Challenges: {', '.join(perspective.potential_challenges[:3])}")
            print(f"   • Success Criteria: {', '.join(perspective.success_criteria[:3])}")
            print(f"   • Estimated Effort: {perspective.estimated_effort:.2f}")
            print(f"   • Confidence Level: {perspective.confidence_level:.2f}")
        
        print(f"\n" + "="*50)
        print("📝 COMPREHENSIVE SUMMARY")
        print("="*50)
        print(analyzer.get_perspective_summary(detailed_analysis))
        
        print("\n✅ Multi-perspective intent analysis test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_intent_analyzer()) 