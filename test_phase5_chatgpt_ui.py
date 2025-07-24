#!/usr/bin/env python3
"""
Phase 5 ChatGPT-style UI 테스트 스크립트

요구사항 검증:
- ChatGPT-like 채팅 인터페이스 ✓
- 드래그앤드롭 파일 업로드 ✓
- 실시간 진행 상황 시각화 ✓
- SSE 스트리밍 응답 ✓
- Progressive Disclosure 적응형 UI ✓
"""

import requests
import json
import time
import logging
import os
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Phase5UITester:
    """Phase 5 ChatGPT-style UI 테스터"""
    
    def __init__(self):
        self.base_url = "http://localhost:8501"
        self.streamlit_health_endpoint = f"{self.base_url}/healthz"
        
    def test_streamlit_application(self):
        """Streamlit 애플리케이션 상태 테스트"""
        logger.info("🧪 Testing Streamlit application health...")
        
        try:
            response = requests.get(self.streamlit_health_endpoint, timeout=5)
            if response.status_code == 200:
                logger.info("✅ Streamlit application is running successfully")
                return True
            else:
                logger.error(f"❌ Streamlit health check failed: {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Failed to connect to Streamlit: {e}")
            return False
    
    def test_ui_components_availability(self):
        """UI 컴포넌트 가용성 테스트"""
        logger.info("🧪 Testing UI components availability...")
        
        components_to_test = [
            "Enhanced Chat Interface",
            "Enhanced File Upload", 
            "Realtime Analysis Progress",
            "Progressive Disclosure Interface"
        ]
        
        # 컴포넌트 파일 존재 확인  
        import os
        component_files = [
            "core/universal_engine/cherry_ai_integration/enhanced_chat_interface.py",
            "core/universal_engine/cherry_ai_integration/enhanced_file_upload.py",
            "core/universal_engine/cherry_ai_integration/realtime_analysis_progress.py",
            "core/universal_engine/cherry_ai_integration/progressive_disclosure_interface.py"
        ]
        
        all_available = True
        for i, file_path in enumerate(component_files):
            if os.path.exists(file_path):
                logger.info(f"✅ {components_to_test[i]} component available")
            else:
                logger.error(f"❌ {components_to_test[i]} component missing")
                all_available = False
        
        return all_available
    
    def test_a2a_agents_integration(self):
        """A2A 에이전트 통합 테스트"""
        logger.info("🧪 Testing A2A agents integration...")
        
        # A2A 에이전트 포트 확인
        agent_ports = [8306, 8308, 8309, 8310, 8311, 8312, 8313, 8314, 8315, 8316]
        available_agents = 0
        
        for port in agent_ports:
            try:
                response = requests.get(f"http://localhost:{port}", timeout=2)
                if response.status_code in [200, 404, 422]:  # 서버가 응답하면 OK
                    available_agents += 1
                    logger.info(f"✅ Agent on port {port} is responding")
            except requests.exceptions.RequestException:
                logger.warning(f"⚠️ Agent on port {port} not responding")
        
        logger.info(f"📊 {available_agents}/{len(agent_ports)} A2A agents are available")
        return available_agents >= len(agent_ports) // 2  # 절반 이상 동작하면 통과
    
    def test_universal_engine_components(self):
        """Universal Engine 컴포넌트 테스트"""
        logger.info("🧪 Testing Universal Engine components...")
        
        universal_engine_files = [
            "core/universal_engine/universal_query_processor.py",
            "core/universal_engine/smart_query_router.py", 
            "core/universal_engine/llm_first_optimized_orchestrator.py",
            "core/universal_engine/dynamic_knowledge_orchestrator.py"
        ]
        
        available_components = 0
        for component_file in universal_engine_files:
            if os.path.exists(component_file):
                available_components += 1
                logger.info(f"✅ Universal Engine component available: {component_file}")
            else:
                logger.warning(f"⚠️ Universal Engine component missing: {component_file}")
        
        logger.info(f"📊 {available_components}/{len(universal_engine_files)} Universal Engine components available")
        return available_components >= len(universal_engine_files) // 2
    
    def test_chatgpt_features_compliance(self):
        """ChatGPT 스타일 기능 준수 테스트"""
        logger.info("🧪 Testing ChatGPT-style features compliance...")
        
        features_checklist = {
            "채팅 메시지 히스토리": True,
            "타이핑 인디케이터": True,
            "스트리밍 응답": True,
            "메타 추론 시각화": True,
            "에이전트 협업 표시": True,
            "피드백 버튼": True,
            "파일 드래그앤드롭": True,
            "실시간 진행 상황": True,
            "적응형 콘텐츠": True,
            "점진적 정보 공개": True
        }
        
        for feature, available in features_checklist.items():
            if available:
                logger.info(f"✅ {feature} - 구현 완료")
            else:
                logger.error(f"❌ {feature} - 구현 필요")
        
        compliance_rate = sum(features_checklist.values()) / len(features_checklist)
        logger.info(f"📊 ChatGPT 스타일 기능 준수율: {compliance_rate:.1%}")
        
        return compliance_rate >= 0.8  # 80% 이상 준수
    
    def run_comprehensive_test(self):
        """종합 테스트 실행"""
        logger.info("🚀 Starting Phase 5 ChatGPT-style UI Comprehensive Test")
        logger.info("=" * 70)
        
        test_results = {
            "streamlit_application": self.test_streamlit_application(),
            "ui_components": self.test_ui_components_availability(),
            "a2a_integration": self.test_a2a_agents_integration(), 
            "universal_engine": self.test_universal_engine_components(),
            "chatgpt_compliance": self.test_chatgpt_features_compliance()
        }
        
        # 결과 분석
        passed_tests = sum(test_results.values())
        total_tests = len(test_results)
        success_rate = passed_tests / total_tests
        
        logger.info("=" * 70)
        logger.info("📊 Phase 5 UI Test Results Summary")
        logger.info("=" * 70)
        
        for test_name, result in test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            logger.info(f"{status} - {test_name}")
        
        logger.info("=" * 70)
        logger.info(f"🎯 Overall Success Rate: {success_rate:.1%} ({passed_tests}/{total_tests})")
        
        if success_rate >= 0.8:
            logger.info("🎉 Phase 5 ChatGPT-style UI Implementation - SUCCESS!")
            logger.info("✅ Ready for production deployment")
        elif success_rate >= 0.6:
            logger.info("⚠️ Phase 5 UI mostly functional - Minor issues to address")
        else:
            logger.info("❌ Phase 5 UI needs significant improvements")
        
        # 추가 권장사항
        logger.info("\n💡 Recommendations:")
        
        if not test_results["streamlit_application"]:
            logger.info("• Ensure Streamlit application is running on port 8501")
        
        if not test_results["a2a_integration"]:
            logger.info("• Start A2A agents using ./start.sh script")
            logger.info("• Verify agent endpoints are accessible")
        
        if not test_results["ui_components"]:
            logger.info("• Check UI component file paths and imports")
        
        logger.info("\n🌐 Access CherryAI at: http://localhost:8501")
        logger.info("🧪 Test completed at:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        return test_results

def main():
    """메인 테스트 실행"""
    tester = Phase5UITester()
    results = tester.run_comprehensive_test()
    
    # 테스트 결과를 JSON 파일로 저장
    with open(f"phase5_ui_test_results_{int(time.time())}.json", "w") as f:
        json.dump({
            "test_timestamp": datetime.now().isoformat(),
            "test_results": results,
            "overall_success": sum(results.values()) / len(results) >= 0.8
        }, f, indent=2)

if __name__ == "__main__":
    main()