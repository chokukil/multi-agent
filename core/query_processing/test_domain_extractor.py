"""
Test script for DomainKnowledgeExtractor
"""

import asyncio
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from core.query_processing.domain_extractor import DomainKnowledgeExtractor

async def test_domain_extractor():
    """Test the DomainKnowledgeExtractor with a complex domain-specific query"""
    
    print("🚀 Testing DomainKnowledgeExtractor...")
    
    # Initialize the extractor
    extractor = DomainKnowledgeExtractor()
    
    # Test query (complex semiconductor manufacturing scenario)
    test_query = """
    LOT 히스토리와 계측값 데이터를 분석해서 공정 이상 여부를 판단하고, 
    이상 원인을 분석해서 기술적 조치 방향을 제안해주세요. 
    또한 향후 유사 문제 예방을 위한 모니터링 시스템도 제안하고,
    품질 관리 체계와 연계하여 전사적 품질 개선 방안을 도출해주세요.
    """
    
    # Sample intent analysis (from previous step)
    intent_analysis = {
        "primary_intent": "Analyze LOT history and measurement data to determine process anomalies and suggest technical corrective actions",
        "query_type": "diagnostic",
        "complexity_level": "complex",
        "urgency_level": "high"
    }
    
    # Sample data context
    data_context = {
        "available_datasets": [
            "ion_implant_3lot_dataset.xlsx",
            "lot_history_data.csv",
            "measurement_data.csv",
            "equipment_logs.csv",
            "quality_control_records.csv"
        ],
        "data_columns": [
            "LOT_ID", "PROCESS_STEP", "MEASUREMENT_VALUE", 
            "SPEC_LIMIT", "TIMESTAMP", "EQUIPMENT_ID", "OPERATOR_ID",
            "QUALITY_GRADE", "DEFECT_TYPE", "INSPECTION_RESULT"
        ],
        "domain": "semiconductor_manufacturing",
        "urgency": "high",
        "stakeholders": ["production_manager", "process_engineer", "quality_assurance", "plant_manager"]
    }
    
    try:
        # Extract comprehensive domain knowledge
        print(f"📋 Extracting domain knowledge from complex query...")
        
        domain_knowledge = await extractor.extract_comprehensive_domain_knowledge(
            test_query, intent_analysis, data_context
        )
        
        # Display comprehensive results
        print("\n" + "="*80)
        print("🎯 COMPREHENSIVE DOMAIN KNOWLEDGE EXTRACTION RESULTS")
        print("="*80)
        
        # Domain Taxonomy
        print(f"\n🏢 Domain Taxonomy:")
        print(f"   • Primary Domain: {domain_knowledge.taxonomy.primary_domain.value}")
        print(f"   • Sub-domains: {', '.join(domain_knowledge.taxonomy.sub_domains)}")
        print(f"   • Industry Sector: {domain_knowledge.taxonomy.industry_sector}")
        print(f"   • Business Function: {domain_knowledge.taxonomy.business_function}")
        print(f"   • Technical Area: {domain_knowledge.taxonomy.technical_area}")
        print(f"   • Taxonomy Confidence: {domain_knowledge.taxonomy.confidence_score:.2f}")
        
        # Key Concepts
        print(f"\n🔑 Key Concepts ({len(domain_knowledge.key_concepts)}):")
        for concept_name, concept_item in list(domain_knowledge.key_concepts.items())[:5]:
            print(f"   • {concept_name} [{concept_item.confidence.value}]: {concept_item.explanation}")
            if concept_item.related_items:
                print(f"     Related: {', '.join(concept_item.related_items[:3])}")
        
        # Technical Terms
        print(f"\n🔧 Technical Terms ({len(domain_knowledge.technical_terms)}):")
        for term_name, term_item in list(domain_knowledge.technical_terms.items())[:5]:
            print(f"   • {term_name} [{term_item.confidence.value}]: {term_item.explanation}")
        
        # Methodology Map
        print(f"\n📋 Methodology Map:")
        print(f"   • Standard Methodologies: {', '.join(domain_knowledge.methodology_map.standard_methodologies[:5])}")
        print(f"   • Best Practices: {', '.join(domain_knowledge.methodology_map.best_practices[:5])}")
        print(f"   • Tools & Technologies: {', '.join(domain_knowledge.methodology_map.tools_and_technologies[:5])}")
        print(f"   • Quality Standards: {', '.join(domain_knowledge.methodology_map.quality_standards[:3])}")
        print(f"   • Compliance Requirements: {', '.join(domain_knowledge.methodology_map.compliance_requirements[:3])}")
        
        # Risk Assessment
        print(f"\n⚠️ Risk Assessment:")
        print(f"   • Technical Risks ({len(domain_knowledge.risk_assessment.technical_risks)}): {', '.join(domain_knowledge.risk_assessment.technical_risks[:3])}")
        print(f"   • Business Risks ({len(domain_knowledge.risk_assessment.business_risks)}): {', '.join(domain_knowledge.risk_assessment.business_risks[:3])}")
        print(f"   • Operational Risks ({len(domain_knowledge.risk_assessment.operational_risks)}): {', '.join(domain_knowledge.risk_assessment.operational_risks[:3])}")
        print(f"   • Compliance Risks ({len(domain_knowledge.risk_assessment.compliance_risks)}): {', '.join(domain_knowledge.risk_assessment.compliance_risks[:3])}")
        print(f"   • Mitigation Strategies: {', '.join(domain_knowledge.risk_assessment.mitigation_strategies[:3])}")
        
        # Stakeholders and Metrics
        print(f"\n👥 Stakeholder Map:")
        for stakeholder_type, stakeholders in domain_knowledge.stakeholder_map.items():
            print(f"   • {stakeholder_type.title()}: {', '.join(stakeholders[:5])}")
        
        print(f"\n📊 Success Metrics:")
        for metric in domain_knowledge.success_metrics[:7]:
            print(f"   • {metric}")
        
        # Business Context
        print(f"\n💼 Business Context:")
        print(f"   {domain_knowledge.business_context}")
        
        print(f"\n✅ Overall Extraction Confidence: {domain_knowledge.extraction_confidence:.2f}")
        
        print(f"\n" + "="*50)
        print("📝 DOMAIN KNOWLEDGE SUMMARY")
        print("="*50)
        print(extractor.get_domain_summary(domain_knowledge))
        
        print("\n✅ Domain knowledge extraction test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_domain_extractor()) 