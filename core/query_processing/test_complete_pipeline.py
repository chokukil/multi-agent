"""
Complete Pipeline Test for CherryAI LLM-First Enhancement Phase 1

This test demonstrates the full Phase 1 pipeline integrating all modules:
- IntelligentQueryProcessor (basic query processing)
- MultiPerspectiveIntentAnalyzer (5-perspective intent analysis)
- DomainKnowledgeExtractor (comprehensive domain knowledge)
- AnswerStructurePredictor (optimal answer structure)
- ContextualQueryEnhancer (query optimization)
"""

import asyncio
import sys
import os
import time

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from core.query_processing.intelligent_query_processor import IntelligentQueryProcessor
from core.query_processing.intent_analyzer import MultiPerspectiveIntentAnalyzer
from core.query_processing.domain_extractor import DomainKnowledgeExtractor
from core.query_processing.answer_predictor import AnswerStructurePredictor
from core.query_processing.query_enhancer import ContextualQueryEnhancer

async def test_complete_pipeline():
    """Test the complete Phase 1 pipeline with all modules integrated"""
    
    print("🚀 CHERRYAI LLM-FIRST ENHANCEMENT - PHASE 1 COMPLETE PIPELINE TEST")
    print("="*80)
    
    # Initialize all components
    print("🔧 Initializing all Phase 1 components...")
    basic_processor = IntelligentQueryProcessor()
    intent_analyzer = MultiPerspectiveIntentAnalyzer()
    domain_extractor = DomainKnowledgeExtractor()
    answer_predictor = AnswerStructurePredictor()
    query_enhancer = ContextualQueryEnhancer()
    
    # Complex test query (realistic semiconductor manufacturing scenario)
    test_query = """
    LOT 히스토리와 계측값 데이터를 분석해서 공정 이상 여부를 판단하고, 
    이상 원인을 분석해서 기술적 조치 방향을 제안해주세요. 
    또한 향후 유사 문제 예방을 위한 모니터링 시스템도 제안하고,
    품질 관리 체계와 연계하여 전사적 품질 개선 방안을 도출해주세요.
    반도체 공정 특성을 고려한 통계적 품질 관리 방법론도 함께 제시하고,
    투자 대비 효과(ROI) 분석과 실행 로드맵을 포함해주세요.
    """
    
    # Rich data context
    data_context = {
        "available_datasets": [
            "ion_implant_3lot_dataset.xlsx",
            "lot_history_data.csv", 
            "measurement_data.csv",
            "equipment_logs.csv",
            "quality_control_records.csv",
            "financial_data.csv",
            "roi_analysis_template.xlsx"
        ],
        "data_columns": [
            "LOT_ID", "PROCESS_STEP", "MEASUREMENT_VALUE", 
            "SPEC_LIMIT", "TIMESTAMP", "EQUIPMENT_ID", "OPERATOR_ID",
            "QUALITY_GRADE", "DEFECT_TYPE", "INSPECTION_RESULT",
            "COST_DATA", "INVESTMENT_AMOUNT", "EXPECTED_SAVINGS"
        ],
        "stakeholders": [
            "production_manager", "process_engineer", "quality_assurance", 
            "plant_manager", "cfo", "operations_director", "cto"
        ],
        "business_context": {
            "urgency": "high",
            "business_impact": "critical",
            "budget_available": True,
            "timeline_pressure": "3_months",
            "regulatory_compliance": ["ISO_9001", "SEMI_standards"]
        }
    }
    
    start_time = time.time()
    
    try:
        print(f"\n📋 Processing Complex Query:")
        print(f"   \"{test_query.strip()[:100]}...\"")
        print(f"   Data Context: {len(data_context['available_datasets'])} datasets, {len(data_context['stakeholders'])} stakeholders")
        
        # PHASE 1 PIPELINE EXECUTION
        print(f"\n" + "="*60)
        print("🔄 PHASE 1 PIPELINE EXECUTION")
        print("="*60)
        
        # Step 1: Basic Query Processing
        print(f"\n🔍 Step 1: Basic Query Processing...")
        step1_start = time.time()
        basic_analysis = await basic_processor.process_query(test_query, data_context)
        step1_time = time.time() - step1_start
        print(f"   ✅ Completed in {step1_time:.2f}s - Confidence: {basic_analysis.intent_analysis.confidence_score:.2f}")
        
        # Step 2: Multi-Perspective Intent Analysis
        print(f"\n🧠 Step 2: Multi-Perspective Intent Analysis...")
        step2_start = time.time()
        detailed_intent = await intent_analyzer.analyze_intent_comprehensive(test_query, data_context)
        step2_time = time.time() - step2_start
        print(f"   ✅ Completed in {step2_time:.2f}s - Confidence: {detailed_intent.overall_confidence:.2f}")
        print(f"   📊 Analyzed from {len(detailed_intent.perspectives)} expert perspectives")
        
        # Step 3: Comprehensive Domain Knowledge Extraction
        print(f"\n🎯 Step 3: Comprehensive Domain Knowledge Extraction...")
        step3_start = time.time()
        domain_knowledge = await domain_extractor.extract_comprehensive_domain_knowledge(
            test_query, 
            {
                "primary_intent": detailed_intent.primary_intent,
                "query_type": detailed_intent.query_type.value,
                "complexity_level": detailed_intent.complexity_level.value
            }, 
            data_context
        )
        step3_time = time.time() - step3_start
        print(f"   ✅ Completed in {step3_time:.2f}s - Confidence: {domain_knowledge.extraction_confidence:.2f}")
        print(f"   🏭 Domain: {domain_knowledge.taxonomy.primary_domain.value} | {domain_knowledge.taxonomy.industry_sector}")
        
        # Step 4: Answer Structure Prediction
        print(f"\n📋 Step 4: Answer Structure Prediction...")
        step4_start = time.time()
        answer_structure = await answer_predictor.predict_optimal_structure(
            detailed_intent, domain_knowledge, data_context
        )
        step4_time = time.time() - step4_start
        print(f"   ✅ Completed in {step4_time:.2f}s - Confidence: {answer_structure.confidence_score:.2f}")
        print(f"   📄 Format: {answer_structure.primary_template.format_type.value} | {len(answer_structure.primary_template.sections)} sections")
        
        # Step 5: Contextual Query Enhancement
        print(f"\n🔧 Step 5: Contextual Query Enhancement...")
        step5_start = time.time()
        query_enhancement = await query_enhancer.enhance_query_comprehensively(
            test_query, detailed_intent, domain_knowledge, answer_structure
        )
        step5_time = time.time() - step5_start
        print(f"   ✅ Completed in {step5_time:.2f}s - Confidence: {query_enhancement.optimization_confidence:.2f}")
        print(f"   🎯 Strategy: {query_enhancement.enhancement_strategy.value} | {len(query_enhancement.query_variations)} variations")
        
        total_time = time.time() - start_time
        
        # COMPREHENSIVE RESULTS DISPLAY
        print(f"\n" + "="*80)
        print("🎯 PHASE 1 PIPELINE RESULTS SUMMARY")
        print("="*80)
        
        # Performance Metrics
        print(f"\n⏱️ Performance Metrics:")
        print(f"   • Total Processing Time: {total_time:.2f}s")
        print(f"   • Average Step Time: {total_time/5:.2f}s")
        print(f"   • Steps: Basic({step1_time:.1f}s) → Intent({step2_time:.1f}s) → Domain({step3_time:.1f}s) → Structure({step4_time:.1f}s) → Enhancement({step5_time:.1f}s)")
        
        # Confidence Scores
        print(f"\n✅ Confidence Scores:")
        print(f"   • Basic Processing: {basic_analysis.intent_analysis.confidence_score:.2f}")
        print(f"   • Intent Analysis: {detailed_intent.overall_confidence:.2f}")
        print(f"   • Domain Knowledge: {domain_knowledge.extraction_confidence:.2f}")
        print(f"   • Answer Structure: {answer_structure.confidence_score:.2f}")
        print(f"   • Query Enhancement: {query_enhancement.optimization_confidence:.2f}")
        overall_confidence = (
            basic_analysis.intent_analysis.confidence_score + 
            detailed_intent.overall_confidence + 
            domain_knowledge.extraction_confidence + 
            answer_structure.confidence_score + 
            query_enhancement.optimization_confidence
        ) / 5
        print(f"   📊 OVERALL PIPELINE CONFIDENCE: {overall_confidence:.2f} ⭐")
        
        # Key Insights
        print(f"\n🔍 Key Insights:")
        print(f"   • Query Type: {detailed_intent.query_type.value} ({detailed_intent.complexity_level.value} complexity)")
        print(f"   • Domain: {domain_knowledge.taxonomy.primary_domain.value} → {domain_knowledge.taxonomy.technical_area}")
        print(f"   • Target Audience: {answer_structure.primary_template.target_audience}")
        print(f"   • Estimated Timeline: {detailed_intent.estimated_timeline}")
        print(f"   • Business Priority: {detailed_intent.execution_priority}/10")
        
        # Enhancement Results
        enhanced_query = query_enhancement.primary_enhanced_query
        print(f"\n🔧 Enhancement Results:")
        print(f"   • Enhancement Types: {len(enhanced_query.enhancement_types)}")
        print(f"   • Estimated Improvement: {enhanced_query.estimated_improvement:.0%}")
        print(f"   • Query Length Change: {enhanced_query.enhancement_metadata.get('original_length', 0)} → {enhanced_query.enhancement_metadata.get('enhanced_length', 0)} chars")
        print(f"   • Available Variations: {len(query_enhancement.query_variations)}")
        
        # Final Output Structure
        template = answer_structure.primary_template
        print(f"\n📋 Recommended Answer Structure:")
        print(f"   • Format: {template.format_type.value}")
        print(f"   • Sections: {len(template.sections)} (Priority-ordered)")
        print(f"   • Visualizations: {len(template.required_visualizations)} types")
        print(f"   • Quality Checkpoints: {len(template.quality_checkpoints)}")
        print(f"   • Estimated Completion: {template.estimated_completion_time}")
        
        # Show Enhanced Query
        print(f"\n" + "="*60)
        print("🎯 ENHANCED QUERY OUTPUT")
        print("="*60)
        print(f"\n📝 Original Query ({len(test_query)} chars):")
        print(f'   "{test_query.strip()}"')
        
        print(f"\n🚀 Enhanced Query ({len(enhanced_query.enhanced_query)} chars):")
        print(f'   "{enhanced_query.enhanced_query}"')
        
        print(f"\n💡 Enhancement Reasoning:")
        print(f"   {enhanced_query.enhancement_reasoning}")
        
        # Show Query Variations
        if query_enhancement.query_variations:
            print(f"\n🔄 Query Variations ({len(query_enhancement.query_variations)}):")
            for i, variation in enumerate(query_enhancement.query_variations, 1):
                print(f"   {i}. {variation.variation_type.title().replace('_', ' ')} (Priority: {variation.relative_priority}/10)")
                print(f"      Use Case: {variation.target_use_case}")
                print(f'      Query: "{variation.variation_query[:100]}..."')
        
        # Success Summary
        print(f"\n" + "="*80)
        print("🎉 PHASE 1 PIPELINE COMPLETION SUMMARY")
        print("="*80)
        print(f"✅ Successfully processed complex {detailed_intent.query_type.value} query")
        print(f"✅ Achieved {overall_confidence:.0%} overall pipeline confidence")
        print(f"✅ Generated {len(query_enhancement.query_variations) + 1} optimized query variants")
        print(f"✅ Predicted comprehensive {template.format_type.value} with {len(template.sections)} sections")
        print(f"✅ Identified {len(domain_knowledge.key_concepts)} key concepts and {len(domain_knowledge.technical_terms)} technical terms")
        print(f"✅ Analyzed from {len(detailed_intent.perspectives)} expert perspectives")
        print(f"✅ Estimated {enhanced_query.estimated_improvement:.0%} improvement from enhancement")
        
        print(f"\n🎯 READY FOR PHASE 2: Knowledge-Aware Orchestration")
        print(f"   The enhanced query and comprehensive analysis are ready for")
        print(f"   intelligent AI agent coordination in Phase 2!")
        
        print(f"\n✅ Phase 1 Complete Pipeline Test completed successfully!")
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_complete_pipeline()) 