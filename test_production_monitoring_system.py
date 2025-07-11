#!/usr/bin/env python3
"""
🧪 Production Monitoring System Comprehensive Test

프로덕션 모니터링 시스템 종합 테스트
- 통합 알림 시스템 테스트
- 시스템 건강성 체커 테스트
- 로그 분석 시스템 테스트
- 성능 모니터링 통합 테스트
- 전체 시스템 통합 테스트

Author: CherryAI Production Team
"""

import asyncio
import os
import sys
import time
import json
import tempfile
import threading
from datetime import datetime, timedelta
from pathlib import Path

# 프로젝트 루트 추가
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# 테스트 대상 시스템들
try:
    from core.integrated_alert_system import get_integrated_alert_system, AlertSeverity, AlertCategory
    from core.system_health_checker import get_system_health_checker
    from core.enhanced_log_analyzer import get_enhanced_log_analyzer
    from core.performance_monitor import PerformanceMonitor
    from core.performance_optimizer import get_performance_optimizer
    SYSTEMS_AVAILABLE = True
    print("✅ 모든 모니터링 시스템 로드 성공")
except ImportError as e:
    SYSTEMS_AVAILABLE = False
    print(f"❌ 모니터링 시스템 로드 실패: {e}")


class ProductionMonitoringSystemTest:
    """프로덕션 모니터링 시스템 테스트"""
    
    def __init__(self):
        self.test_results = {}
        self.test_start_time = datetime.now()
        
        if SYSTEMS_AVAILABLE:
            self.alert_system = get_integrated_alert_system()
            self.health_checker = get_system_health_checker()
            self.log_analyzer = get_enhanced_log_analyzer()
            self.performance_optimizer = get_performance_optimizer()
            self.performance_monitor = PerformanceMonitor()
        else:
            self.alert_system = None
            self.health_checker = None
            self.log_analyzer = None
            self.performance_optimizer = None
            self.performance_monitor = None
    
    def run_comprehensive_test(self):
        """종합 테스트 실행"""
        print("🧪 프로덕션 모니터링 시스템 종합 테스트")
        print(f"⏰ 시작 시간: {self.test_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        if not SYSTEMS_AVAILABLE:
            print("❌ 모니터링 시스템을 사용할 수 없습니다.")
            return
        
        # 테스트 케이스들
        test_cases = [
            ("통합 알림 시스템", self.test_integrated_alert_system),
            ("시스템 건강성 체커", self.test_system_health_checker),
            ("로그 분석 시스템", self.test_log_analyzer),
            ("성능 모니터링", self.test_performance_monitoring),
            ("시스템 통합", self.test_system_integration),
        ]
        
        for test_name, test_func in test_cases:
            print(f"\n📊 {test_name} 테스트")
            print("-" * 40)
            
            try:
                start_time = time.time()
                result = test_func()
                execution_time = time.time() - start_time
                
                result["execution_time"] = execution_time
                self.test_results[test_name] = result
                
                if result["success"]:
                    print(f"✅ {test_name} 테스트 성공 ({execution_time:.2f}초)")
                else:
                    print(f"❌ {test_name} 테스트 실패: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                print(f"❌ {test_name} 테스트 중 예외 발생: {e}")
                self.test_results[test_name] = {
                    "success": False,
                    "error": str(e),
                    "execution_time": 0
                }
        
        # 결과 요약
        self.generate_test_report()
    
    def test_integrated_alert_system(self) -> dict:
        """통합 알림 시스템 테스트"""
        if not self.alert_system:
            return {"success": False, "error": "Alert system not available"}
        
        try:
            # 1. 모니터링 시작
            self.alert_system.start_monitoring()
            time.sleep(2)  # 시작 대기
            
            # 2. 시스템 상태 확인
            status = self.alert_system.get_system_status()
            assert status["monitoring_active"], "모니터링이 활성화되지 않음"
            
            # 3. 활성 알림 확인
            active_alerts = self.alert_system.get_active_alerts()
            print(f"   현재 활성 알림: {len(active_alerts)}개")
            
            # 4. 알림 이력 확인
            alert_history = self.alert_system.get_alert_history(hours=1)
            print(f"   최근 1시간 알림: {len(alert_history)}개")
            
            # 5. 성능 권장사항 확인
            if hasattr(self.alert_system, 'performance_optimizer') and self.alert_system.performance_optimizer:
                recommendations = self.alert_system.performance_optimizer.get_performance_recommendations()
                print(f"   성능 권장사항: {len(recommendations)}개")
            
            return {
                "success": True,
                "active_alerts": len(active_alerts),
                "alert_history": len(alert_history),
                "monitoring_active": status["monitoring_active"],
                "enabled_channels": status["enabled_channels"],
                "enabled_rules": status["enabled_rules"]
            }
        
        except Exception as e:
            return {"success": False, "error": str(e)}
        finally:
            # 정리
            if self.alert_system:
                self.alert_system.stop_monitoring()
    
    def test_system_health_checker(self) -> dict:
        """시스템 건강성 체커 테스트"""
        if not self.health_checker:
            return {"success": False, "error": "Health checker not available"}
        
        try:
            # 1. 건강성 체크 실행
            print("   시스템 건강성 체크 실행 중...")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            report = loop.run_until_complete(self.health_checker.check_system_health())
            loop.close()
            
            # 2. 보고서 검증
            assert report is not None, "건강성 보고서가 생성되지 않음"
            assert hasattr(report, 'overall_score'), "전체 점수가 없음"
            assert hasattr(report, 'component_results'), "컴포넌트 결과가 없음"
            
            print(f"   전체 건강성 점수: {report.overall_score:.1f}%")
            print(f"   체크된 컴포넌트: {len(report.component_results)}개")
            print(f"   심각한 문제: {len(report.critical_issues)}개")
            print(f"   권장사항: {len(report.recommendations)}개")
            
            # 3. 모니터링 상태 확인
            monitoring_status = self.health_checker.get_monitoring_status()
            
            return {
                "success": True,
                "overall_score": report.overall_score,
                "overall_status": report.overall_status.value,
                "components_checked": len(report.component_results),
                "critical_issues": len(report.critical_issues),
                "recommendations": len(report.recommendations),
                "monitoring_status": monitoring_status
            }
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def test_log_analyzer(self) -> dict:
        """로그 분석 시스템 테스트"""
        if not self.log_analyzer:
            return {"success": False, "error": "Log analyzer not available"}
        
        try:
            # 1. 테스트 로그 파일 생성
            test_log_dir = Path("test_logs")
            test_log_dir.mkdir(exist_ok=True)
            
            test_log_file = test_log_dir / "test_monitoring.log"
            
            # 다양한 로그 레벨과 패턴 생성
            test_logs = [
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - INFO - test - 시스템 시작",
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - ERROR - test - 메모리 부족 오류 발생",
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - WARNING - test - CPU 사용률 높음",
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - ERROR - test - Agent failed to respond",
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - CRITICAL - test - DatabaseError: connection timeout",
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - INFO - test - 처리 완료"
            ]
            
            with open(test_log_file, 'w', encoding='utf-8') as f:
                for log_line in test_logs:
                    f.write(log_line + "\n")
            
            # 2. 로그 분석기 설정
            analyzer = get_enhanced_log_analyzer(log_directory=str(test_log_dir))
            
            # 3. 로그 처리
            analyzer._process_log_file(str(test_log_file))
            
            # 4. 분석 보고서 생성
            report = analyzer.generate_analysis_report(hours=1)
            
            print(f"   처리된 로그 엔트리: {report.total_entries}개")
            print(f"   레벨별 분포: {report.entries_by_level}")
            print(f"   패턴 매치: {len(report.pattern_matches)}개")
            print(f"   이상 징후: {len(report.anomalies)}개")
            
            # 5. 모니터링 상태 확인
            monitoring_status = analyzer.get_monitoring_status()
            
            # 6. 정리
            test_log_file.unlink(missing_ok=True)
            test_log_dir.rmdir()
            
            return {
                "success": True,
                "total_entries": report.total_entries,
                "entries_by_level": report.entries_by_level,
                "pattern_matches": len(report.pattern_matches),
                "anomalies": len(report.anomalies),
                "recommendations": len(report.recommendations),
                "monitoring_status": monitoring_status
            }
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def test_performance_monitoring(self) -> dict:
        """성능 모니터링 테스트"""
        try:
            results = {}
            
            # 1. 성능 최적화기 테스트
            if self.performance_optimizer:
                print("   성능 최적화기 테스트...")
                
                # 메모리 최적화 테스트
                memory_result = self.performance_optimizer.optimize_memory()
                results["memory_optimization"] = {
                    "success": memory_result.success,
                    "improvement_percent": memory_result.improvement_percent
                }
                print(f"   메모리 최적화: {memory_result.improvement_percent:.1f}% 개선")
                
                # 성능 권장사항 테스트
                recommendations = self.performance_optimizer.get_performance_recommendations()
                results["recommendations"] = len(recommendations)
                print(f"   성능 권장사항: {len(recommendations)}개")
            
            # 2. 성능 모니터 테스트
            if self.performance_monitor:
                print("   성능 모니터 테스트...")
                
                # 모니터링 시작
                self.performance_monitor.start_monitoring()
                time.sleep(3)  # 데이터 수집 대기
                
                # 성능 요약 확인
                try:
                    summary = self.performance_monitor.get_performance_summary()
                    results["performance_summary"] = summary
                    print(f"   모니터링 상태: {summary.get('monitoring_active', 'Unknown')}")
                except Exception as e:
                    print(f"   성능 요약 오류: {e}")
                    results["performance_summary"] = {"error": str(e)}
                
                # 모니터링 중지
                self.performance_monitor.stop_monitoring()
            
            return {
                "success": True,
                **results
            }
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def test_system_integration(self) -> dict:
        """시스템 통합 테스트"""
        try:
            integration_results = {}
            
            # 1. 모든 시스템 동시 시작
            print("   모든 시스템 동시 시작...")
            systems_started = []
            
            if self.alert_system:
                self.alert_system.start_monitoring()
                systems_started.append("alert_system")
            
            if self.health_checker:
                self.health_checker.start_monitoring()
                systems_started.append("health_checker")
            
            if self.log_analyzer:
                self.log_analyzer.start_monitoring()
                systems_started.append("log_analyzer")
            
            if self.performance_monitor:
                self.performance_monitor.start_monitoring()
                systems_started.append("performance_monitor")
            
            integration_results["systems_started"] = systems_started
            print(f"   시작된 시스템: {len(systems_started)}개")
            
            # 2. 시스템 간 상호작용 테스트
            time.sleep(5)  # 시스템 안정화 대기
            
            # 3. 전체 상태 수집
            if self.alert_system:
                alert_status = self.alert_system.get_system_status()
                integration_results["alert_system_status"] = alert_status
            
            if self.health_checker:
                health_status = self.health_checker.get_monitoring_status()
                integration_results["health_checker_status"] = health_status
            
            if self.log_analyzer:
                log_status = self.log_analyzer.get_monitoring_status()
                integration_results["log_analyzer_status"] = log_status
            
            # 4. 모든 시스템 중지
            print("   모든 시스템 중지...")
            if self.alert_system:
                self.alert_system.stop_monitoring()
            
            if self.health_checker:
                self.health_checker.stop_monitoring()
            
            if self.log_analyzer:
                self.log_analyzer.stop_monitoring()
            
            if self.performance_monitor:
                self.performance_monitor.stop_monitoring()
            
            return {
                "success": True,
                **integration_results
            }
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def generate_test_report(self):
        """테스트 보고서 생성"""
        print("\n" + "=" * 60)
        print("📊 프로덕션 모니터링 시스템 테스트 결과")
        print("=" * 60)
        
        total_tests = len(self.test_results)
        successful_tests = sum(1 for result in self.test_results.values() if result.get("success", False))
        
        print(f"✅ 성공한 테스트: {successful_tests}/{total_tests}")
        print(f"⏱️ 총 실행 시간: {(datetime.now() - self.test_start_time).total_seconds():.2f}초")
        print()
        
        # 각 테스트 결과 상세
        for test_name, result in self.test_results.items():
            print(f"📈 {test_name}:")
            if result.get("success", False):
                print(f"   ✅ 성공 (실행 시간: {result.get('execution_time', 0):.2f}초)")
                
                # 세부 결과 표시
                if test_name == "통합 알림 시스템":
                    print(f"   - 활성 알림: {result.get('active_alerts', 0)}개")
                    print(f"   - 모니터링 활성: {result.get('monitoring_active', False)}")
                    print(f"   - 활성 채널: {result.get('enabled_channels', 0)}개")
                
                elif test_name == "시스템 건강성 체커":
                    print(f"   - 전체 점수: {result.get('overall_score', 0):.1f}%")
                    print(f"   - 상태: {result.get('overall_status', 'unknown')}")
                    print(f"   - 체크된 컴포넌트: {result.get('components_checked', 0)}개")
                
                elif test_name == "로그 분석 시스템":
                    print(f"   - 처리된 엔트리: {result.get('total_entries', 0)}개")
                    print(f"   - 패턴 매치: {result.get('pattern_matches', 0)}개")
                    print(f"   - 이상 징후: {result.get('anomalies', 0)}개")
                
                elif test_name == "성능 모니터링":
                    if "memory_optimization" in result:
                        mem_opt = result["memory_optimization"]
                        print(f"   - 메모리 최적화: {mem_opt.get('improvement_percent', 0):.1f}%")
                    print(f"   - 권장사항: {result.get('recommendations', 0)}개")
                
                elif test_name == "시스템 통합":
                    systems_started = result.get('systems_started', [])
                    print(f"   - 시작된 시스템: {len(systems_started)}개")
                    for system in systems_started:
                        print(f"     • {system}")
            else:
                print(f"   ❌ 실패: {result.get('error', 'Unknown error')}")
        
        # 종합 평가
        success_rate = (successful_tests / total_tests) * 100
        print(f"\n🎯 종합 성공률: {success_rate:.1f}%")
        
        if success_rate >= 90:
            print("🎉 프로덕션 모니터링 시스템이 우수한 상태입니다!")
        elif success_rate >= 70:
            print("✅ 프로덕션 모니터링 시스템이 양호한 상태입니다.")
        else:
            print("⚠️ 프로덕션 모니터링 시스템에 개선이 필요합니다.")
        
        # 결과 파일 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = f"production_monitoring_test_results_{timestamp}.json"
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(self.test_results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n📁 상세 결과가 저장되었습니다: {result_file}")


def main():
    """메인 테스트 실행"""
    test_suite = ProductionMonitoringSystemTest()
    test_suite.run_comprehensive_test()


if __name__ == "__main__":
    main() 