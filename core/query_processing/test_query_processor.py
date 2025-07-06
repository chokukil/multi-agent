"""
Test script for IntelligentQueryProcessor
"""

import asyncio
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from core.query_processing.intelligent_query_processor import IntelligentQueryProcessor

async def test_query_processor():
    """Test the IntelligentQueryProcessor with a sample query"""
    
    print("🚀 Testing IntelligentQueryProcessor...")
    
    # Initialize the processor
    processor = IntelligentQueryProcessor()
    
    # Test query (similar to the user's example)
    test_query = """
    LOT 히스토리와 계측값 데이터를 분석해서 공정 이상 여부를 판단하고, 
    이상 원인을 분석해서 기술적 조치 방향을 제안해주세요.
    """
    
    # Sample data context
    data_context = {
        "available_datasets": [
            "ion_implant_3lot_dataset.xlsx",
            "lot_history_data.csv",
            "measurement_data.csv"
        ],
        "data_columns": [
            "LOT_ID", "PROCESS_STEP", "MEASUREMENT_VALUE", 
            "SPEC_LIMIT", "TIMESTAMP", "EQUIPMENT_ID"
        ],
        "domain": "semiconductor_manufacturing"
    }
    
    try:
        # Process the query
        print(f"📋 Processing query: {test_query.strip()}")
        
        enhanced_query = await processor.process_query(test_query, data_context)
        
        # Display results
        print("\n" + "="*60)
        print("🎯 QUERY PROCESSING RESULTS")
        print("="*60)
        
        print(f"\n📊 Intent Analysis:")
        print(f"  • Primary Intent: {enhanced_query.intent_analysis.primary_intent}")
        print(f"  • Query Type: {enhanced_query.intent_analysis.query_type.value}")
        print(f"  • Complexity: {enhanced_query.intent_analysis.complexity_score:.2f}")
        print(f"  • Urgency: {enhanced_query.intent_analysis.urgency_level:.2f}")
        print(f"  • Confidence: {enhanced_query.intent_analysis.confidence_score:.2f}")
        
        print(f"\n🎯 Domain Knowledge:")
        print(f"  • Domain: {enhanced_query.domain_knowledge.domain_type.value}")
        print(f"  • Key Concepts: {', '.join(enhanced_query.domain_knowledge.key_concepts[:5])}")
        print(f"  • Required Expertise: {', '.join(enhanced_query.domain_knowledge.required_expertise[:3])}")
        print(f"  • Business Context: {enhanced_query.domain_knowledge.business_context}")
        
        print(f"\n📋 Expected Answer Structure:")
        print(f"  • Format: {enhanced_query.answer_structure.expected_format.value}")
        print(f"  • Key Sections: {', '.join(enhanced_query.answer_structure.key_sections[:5])}")
        print(f"  • Required Visualizations: {', '.join(enhanced_query.answer_structure.required_visualizations[:3])}")
        
        print(f"\n🔄 Enhanced Queries:")
        for i, eq in enumerate(enhanced_query.enhanced_queries[:3], 1):
            print(f"  {i}. {eq}")
        
        print(f"\n🚀 Execution Strategy: {enhanced_query.execution_strategy}")
        
        print(f"\n📝 Query Summary:")
        print(processor.get_query_summary(enhanced_query))
        
        print("\n✅ Query processing test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_query_processor()) 