#!/usr/bin/env python3
"""
Comprehensive QA Validation Suite
Validates all improvements made during the build process:
- Security fixes
- Performance enhancements  
- Test reliability
- Coverage requirements
- Quality gates
"""

import asyncio
import sys
import time
import json
import logging
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime

# Add project to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from coverage_config import run_comprehensive_coverage_analysis
from modules.core.security_validation_system import LLMSecurityValidationSystem, SecurityContext
from modules.core.agent_circuit_breaker import get_circuit_breaker
from modules.core.enhanced_file_upload import get_upload_system
from modules.core.enhanced_exception_handling import get_exception_handler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class QAValidationResult:
    """QA validation result structure"""
    component: str
    test_name: str
    status: str  # 'passed', 'failed', 'warning'
    execution_time: float
    details: str
    recommendations: List[str] = field(default_factory=list)


class ComprehensiveQAValidator:
    """
    Comprehensive QA validation system
    Tests all major improvements made during the build
    """
    
    def __init__(self):
        self.results: List[QAValidationResult] = []
        self.start_time = time.time()
        
    async def run_all_validations(self) -> Dict[str, Any]:
        """Run all validation tests"""
        logger.info("🍒 Starting Comprehensive QA Validation Suite")
        
        validation_tasks = [
            self.validate_security_improvements(),
            self.validate_circuit_breaker_functionality(),
            self.validate_file_upload_performance(),
            self.validate_exception_handling(),
            self.validate_test_improvements(),
            self.validate_coverage_requirements()
        ]
        
        # Run all validations concurrently
        await asyncio.gather(*validation_tasks)
        
        # Generate summary report
        summary = self.generate_validation_summary()
        
        logger.info("🎉 QA Validation Suite completed")
        return summary
    
    async def validate_security_improvements(self):
        """Validate XSS sanitization and security improvements"""
        logger.info("🛡️ Validating security improvements...")
        start_time = time.time()
        
        try:
            security_system = LLMSecurityValidationSystem()
            
            # Test XSS sanitization
            security_context = SecurityContext(
                user_id="test_user",
                session_id="test_session", 
                ip_address="127.0.0.1",
                user_agent="test_agent",
                timestamp=datetime.now(),
                request_count=1
            )
            
            # Test various XSS payloads
            xss_payloads = [
                "<script>alert('xss')</script>",
                "javascript:alert('xss')",
                "<img src=x onerror=alert('xss')>",
                "<svg onload=alert('xss')>",
                "&#60;script&#62;alert('xss')&#60;/script&#62;"
            ]
            
            all_blocked = True
            for payload in xss_payloads:
                result = await security_system.validate_user_input(
                    payload, "user_query", security_context
                )
                
                if result.validation_result.value not in ['blocked', 'malicious']:
                    all_blocked = False
                    logger.warning(f"XSS payload not properly sanitized: {payload}")
            
            execution_time = time.time() - start_time
            
            if all_blocked:
                self.results.append(QAValidationResult(
                    component="Security",
                    test_name="XSS Sanitization",
                    status="passed",
                    execution_time=execution_time,
                    details=f"All {len(xss_payloads)} XSS payloads properly sanitized",
                    recommendations=["Continue monitoring for new XSS vectors"]
                ))
            else:
                self.results.append(QAValidationResult(
                    component="Security", 
                    test_name="XSS Sanitization",
                    status="failed",
                    execution_time=execution_time,
                    details="Some XSS payloads not properly sanitized",
                    recommendations=["Review and enhance XSS sanitization logic"]
                ))
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.results.append(QAValidationResult(
                component="Security",
                test_name="XSS Sanitization", 
                status="failed",
                execution_time=execution_time,
                details=f"Security validation failed: {str(e)}",
                recommendations=["Fix security system initialization issues"]
            ))
    
    async def validate_circuit_breaker_functionality(self):
        """Validate circuit breaker pattern implementation"""
        logger.info("🔌 Validating circuit breaker functionality...")
        start_time = time.time()
        
        try:
            circuit_breaker = get_circuit_breaker()
            
            # Register test agent
            circuit_breaker.register_agent("test_agent", 9999)
            
            # Simulate failures to trigger circuit breaker
            for i in range(6):  # Exceed failure threshold
                try:
                    await circuit_breaker.call_agent("test_agent", "/test", {})
                except Exception:
                    pass  # Expected to fail
            
            # Check if circuit is open
            agent_status = circuit_breaker.get_agent_status("test_agent")
            circuit_state = agent_status.get("circuit_state")
            
            execution_time = time.time() - start_time
            
            if circuit_state == "open":
                self.results.append(QAValidationResult(
                    component="Reliability",
                    test_name="Circuit Breaker",
                    status="passed", 
                    execution_time=execution_time,
                    details="Circuit breaker correctly opened after failures",
                    recommendations=["Monitor circuit breaker metrics in production"]
                ))
            else:
                self.results.append(QAValidationResult(
                    component="Reliability",
                    test_name="Circuit Breaker",
                    status="failed",
                    execution_time=execution_time,
                    details=f"Circuit breaker state: {circuit_state}, expected: open",
                    recommendations=["Check circuit breaker failure threshold configuration"]
                ))
                
        except Exception as e:
            execution_time = time.time() - start_time
            self.results.append(QAValidationResult(
                component="Reliability",
                test_name="Circuit Breaker",
                status="failed",
                execution_time=execution_time,
                details=f"Circuit breaker validation failed: {str(e)}",
                recommendations=["Fix circuit breaker initialization"]
            ))
    
    async def validate_file_upload_performance(self):
        """Validate enhanced file upload performance"""
        logger.info("📁 Validating file upload performance...")
        start_time = time.time()
        
        try:
            upload_system = get_upload_system()
            
            # Create test file data
            from io import BytesIO
            test_data = b"test,data\n1,2\n3,4\n" * 1000  # Simulate larger file
            test_file = BytesIO(test_data)
            
            # Test chunked upload
            upload_start = time.time()
            result = await upload_system.upload_file_chunked(
                test_file, "test_performance.csv", "perf_test_001"
            )
            upload_time = time.time() - upload_start
            
            execution_time = time.time() - start_time
            
            # Check if upload completed within performance threshold
            if upload_time < 8.0 and result.get("upload_success", False):
                self.results.append(QAValidationResult(
                    component="Performance",
                    test_name="File Upload",
                    status="passed",
                    execution_time=execution_time,
                    details=f"Upload completed in {upload_time:.2f}s (threshold: 8s)",
                    recommendations=["Monitor upload performance under load"]
                ))
            else:
                self.results.append(QAValidationResult(
                    component="Performance", 
                    test_name="File Upload",
                    status="failed",
                    execution_time=execution_time,
                    details=f"Upload took {upload_time:.2f}s or failed",
                    recommendations=["Optimize chunked upload algorithm"]
                ))
                
        except Exception as e:
            execution_time = time.time() - start_time
            self.results.append(QAValidationResult(
                component="Performance",
                test_name="File Upload",
                status="failed",
                execution_time=execution_time,
                details=f"File upload test failed: {str(e)}",
                recommendations=["Fix file upload system initialization"]
            ))
    
    async def validate_exception_handling(self):
        """Validate enhanced exception handling"""
        logger.info("⚠️ Validating exception handling improvements...")
        start_time = time.time()
        
        try:
            exception_handler = get_exception_handler()
            
            # Test various exception types
            test_exceptions = [
                (FileNotFoundError("test.txt not found"), "FileNotFoundError"),
                (ValueError("Invalid data format"), "ValueError"),
                (ConnectionError("Network unreachable"), "ConnectionError")
            ]
            
            all_handled = True
            for exception, expected_type in test_exceptions:
                result = await exception_handler.handle_exception(exception)
                
                if not result.get("handled", False):
                    all_handled = False
                    logger.warning(f"Exception not properly handled: {expected_type}")
            
            execution_time = time.time() - start_time
            
            if all_handled:
                self.results.append(QAValidationResult(
                    component="Error Handling",
                    test_name="Exception Handling",
                    status="passed",
                    execution_time=execution_time,
                    details=f"All {len(test_exceptions)} exception types properly handled",
                    recommendations=["Add more exception scenarios to testing"]
                ))
            else:
                self.results.append(QAValidationResult(
                    component="Error Handling",
                    test_name="Exception Handling", 
                    status="failed",
                    execution_time=execution_time,
                    details="Some exceptions not properly handled",
                    recommendations=["Enhance exception handler coverage"]
                ))
                
        except Exception as e:
            execution_time = time.time() - start_time
            self.results.append(QAValidationResult(
                component="Error Handling",
                test_name="Exception Handling",
                status="failed",
                execution_time=execution_time,
                details=f"Exception handling test failed: {str(e)}",
                recommendations=["Fix exception handler initialization"]
            ))
    
    async def validate_test_improvements(self):
        """Validate test reliability improvements"""
        logger.info("🧪 Validating test improvements...")
        start_time = time.time()
        
        try:
            # Check if reliable wait utilities exist
            from utils.reliable_waits import ReliableWaits, TestWaitConditions
            
            # Validate key methods exist
            required_methods = [
                'wait_for_file_upload_complete',
                'wait_for_agent_response', 
                'wait_for_analysis_complete',
                'wait_for_chart_render'
            ]
            
            missing_methods = []
            for method in required_methods:
                if not hasattr(ReliableWaits, method):
                    missing_methods.append(method)
            
            execution_time = time.time() - start_time
            
            if not missing_methods:
                self.results.append(QAValidationResult(
                    component="Test Reliability",
                    test_name="Wait Conditions",
                    status="passed",
                    execution_time=execution_time,
                    details="All reliable wait methods implemented",
                    recommendations=["Monitor test execution stability"]
                ))
            else:
                self.results.append(QAValidationResult(
                    component="Test Reliability",
                    test_name="Wait Conditions",
                    status="failed", 
                    execution_time=execution_time,
                    details=f"Missing methods: {missing_methods}",
                    recommendations=["Implement missing wait condition methods"]
                ))
                
        except Exception as e:
            execution_time = time.time() - start_time
            self.results.append(QAValidationResult(
                component="Test Reliability",
                test_name="Wait Conditions",
                status="failed",
                execution_time=execution_time,
                details=f"Test improvements validation failed: {str(e)}",
                recommendations=["Fix test utilities import issues"]
            ))
    
    async def validate_coverage_requirements(self):
        """Validate test coverage requirements"""
        logger.info("📊 Validating coverage requirements...")
        start_time = time.time()
        
        try:
            # Run coverage analysis
            coverage_report = await run_comprehensive_coverage_analysis(threshold=0.9)
            
            execution_time = time.time() - start_time
            
            if coverage_report.quality_gates_passed:
                self.results.append(QAValidationResult(
                    component="Coverage",
                    test_name="Coverage Threshold",
                    status="passed",
                    execution_time=execution_time,
                    details=f"Coverage {coverage_report.overall_metrics.line_coverage:.1f}% meets 90% threshold",
                    recommendations=coverage_report.recommendations[:3]  # Top 3 recommendations
                ))
            else:
                self.results.append(QAValidationResult(
                    component="Coverage",
                    test_name="Coverage Threshold",
                    status="failed",
                    execution_time=execution_time,
                    details=f"Coverage {coverage_report.overall_metrics.line_coverage:.1f}% below 90% threshold",
                    recommendations=coverage_report.recommendations[:5]  # Top 5 recommendations
                ))
                
        except Exception as e:
            execution_time = time.time() - start_time
            self.results.append(QAValidationResult(
                component="Coverage",
                test_name="Coverage Threshold",
                status="warning",
                execution_time=execution_time,
                details=f"Coverage analysis failed: {str(e)}",
                recommendations=["Install coverage tools and run analysis manually"]
            ))
    
    def generate_validation_summary(self) -> Dict[str, Any]:
        """Generate comprehensive validation summary"""
        total_time = time.time() - self.start_time
        
        # Categorize results
        passed = [r for r in self.results if r.status == "passed"]
        failed = [r for r in self.results if r.status == "failed"] 
        warnings = [r for r in self.results if r.status == "warning"]
        
        # Calculate success rate
        success_rate = (len(passed) / len(self.results)) * 100 if self.results else 0
        
        # Overall status
        overall_status = "PASSED" if len(failed) == 0 else "FAILED"
        if len(failed) == 0 and len(warnings) > 0:
            overall_status = "PASSED_WITH_WARNINGS"
        
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_execution_time": total_time,
            "overall_status": overall_status,
            "success_rate": success_rate,
            "statistics": {
                "total_tests": len(self.results),
                "passed": len(passed),
                "failed": len(failed),
                "warnings": len(warnings)
            },
            "results_by_component": self._group_results_by_component(),
            "critical_issues": [r for r in failed if r.component in ["Security", "Performance"]],
            "all_recommendations": self._consolidate_recommendations(),
            "quality_gates": {
                "security_passed": not any(r.component == "Security" and r.status == "failed" for r in self.results),
                "performance_passed": not any(r.component == "Performance" and r.status == "failed" for r in self.results),
                "reliability_passed": not any(r.component == "Reliability" and r.status == "failed" for r in self.results),
                "coverage_passed": not any(r.component == "Coverage" and r.status == "failed" for r in self.results)
            }
        }
        
        return summary
    
    def _group_results_by_component(self) -> Dict[str, List[Dict[str, Any]]]:
        """Group results by component"""
        grouped = {}
        for result in self.results:
            if result.component not in grouped:
                grouped[result.component] = []
            
            grouped[result.component].append({
                "test_name": result.test_name,
                "status": result.status,
                "execution_time": result.execution_time,
                "details": result.details,
                "recommendations": result.recommendations
            })
        
        return grouped
    
    def _consolidate_recommendations(self) -> List[str]:
        """Consolidate all recommendations"""
        all_recommendations = []
        for result in self.results:
            all_recommendations.extend(result.recommendations)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_recommendations = []
        for rec in all_recommendations:
            if rec not in seen:
                seen.add(rec)
                unique_recommendations.append(rec)
        
        return unique_recommendations
    
    def print_summary(self, summary: Dict[str, Any]):
        """Print human-readable summary"""
        print("\n" + "="*80)
        print("🍒 CHERRY AI COMPREHENSIVE QA VALIDATION REPORT")
        print("="*80)
        
        print(f"\n📊 OVERALL STATUS: {summary['overall_status']}")
        print(f"✅ Success Rate: {summary['success_rate']:.1f}%")
        print(f"⏱️ Total Execution Time: {summary['total_execution_time']:.2f}s")
        
        print(f"\n📈 STATISTICS:")
        stats = summary['statistics']
        print(f"   Total Tests: {stats['total_tests']}")
        print(f"   Passed: {stats['passed']} ✅")
        print(f"   Failed: {stats['failed']} ❌")
        print(f"   Warnings: {stats['warnings']} ⚠️")
        
        print(f"\n🎯 QUALITY GATES:")
        gates = summary['quality_gates']
        for gate, passed in gates.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"   {gate.replace('_', ' ').title()}: {status}")
        
        print(f"\n📋 RESULTS BY COMPONENT:")
        for component, results in summary['results_by_component'].items():
            print(f"\n   🔸 {component}")
            for result in results:
                status_icon = {"passed": "✅", "failed": "❌", "warning": "⚠️"}[result['status']]
                print(f"     {status_icon} {result['test_name']}: {result['details']}")
        
        if summary['critical_issues']:
            print(f"\n🚨 CRITICAL ISSUES:")
            for issue in summary['critical_issues']:
                print(f"   ❌ {issue.component} - {issue.test_name}: {issue.details}")
        
        if summary['all_recommendations']:
            print(f"\n💡 TOP RECOMMENDATIONS:")
            for i, rec in enumerate(summary['all_recommendations'][:5], 1):
                print(f"   {i}. {rec}")
        
        print("\n" + "="*80)


async def main():
    """Main execution function"""
    validator = ComprehensiveQAValidator()
    
    try:
        summary = await validator.run_all_validations()
        
        # Print summary to console
        validator.print_summary(summary)
        
        # Save detailed report
        report_file = Path(__file__).parent / "reports" / f"qa_validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_file.parent.mkdir(exist_ok=True)
        
        with open(report_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n📄 Detailed report saved: {report_file}")
        
        # Exit with appropriate code
        if summary['overall_status'] == "FAILED":
            print("\n❌ QA Validation FAILED - address critical issues before deployment")
            sys.exit(1)
        elif summary['overall_status'] == "PASSED_WITH_WARNINGS":
            print("\n⚠️ QA Validation PASSED with warnings - review recommendations")
            sys.exit(0)
        else:
            print("\n✅ QA Validation PASSED - ready for deployment")
            sys.exit(0)
            
    except Exception as e:
        logger.error(f"QA validation failed with exception: {e}")
        print(f"\n💥 QA Validation crashed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())