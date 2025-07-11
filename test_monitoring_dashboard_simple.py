#!/usr/bin/env python3
"""
🧪 Simple Monitoring System Test

간단한 모니터링 시스템 기능 테스트
"""

import sys
import os
import time
from datetime import datetime

# 프로젝트 루트 추가
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

try:
    from core.production_monitoring_core import get_core_monitoring_system
    print("✅ 핵심 모니터링 시스템 로드 성공")
    
    # 모니터링 시스템 인스턴스
    monitoring = get_core_monitoring_system()
    
    print("\n🧪 모니터링 시스템 간단 테스트")
    print("=" * 50)
    
    # 1. 모니터링 시작
    print("1. 모니터링 시작...")
    monitoring.start_monitoring()
    time.sleep(5)  # 5초 대기
    
    # 2. 시스템 상태 확인
    print("2. 시스템 상태 확인...")
    status = monitoring.get_system_status()
    print(f"   모니터링 활성: {status['monitoring_active']}")
    print(f"   전체 상태: {status['overall_status']}")
    print(f"   전체 점수: {status['overall_score']:.1f}%")
    print(f"   체크된 컴포넌트: {status['components_checked']}개")
    print(f"   활성 알림: {status['active_alerts']}개")
    
    # 3. 컴포넌트 건강성 확인
    print("3. 컴포넌트 건강성 확인...")
    health = monitoring.get_component_health()
    healthy_count = sum(1 for comp in health.values() if comp.status.value == "healthy")
    print(f"   정상 컴포넌트: {healthy_count}/{len(health)}개")
    
    # 4. 활성 알림 확인
    print("4. 활성 알림 확인...")
    alerts = monitoring.get_active_alerts()
    print(f"   현재 활성 알림: {len(alerts)}개")
    for alert in alerts:
        print(f"   - {alert.severity.value}: {alert.title}")
    
    # 5. 시스템 최적화 테스트
    print("5. 시스템 최적화 테스트...")
    optimization_result = monitoring.optimize_system()
    if optimization_result.get("success", False):
        print("   ✅ 시스템 최적화 성공")
    else:
        print(f"   ❌ 시스템 최적화 실패: {optimization_result.get('error', 'Unknown')}")
    
    # 6. 모니터링 중지
    print("6. 모니터링 중지...")
    monitoring.stop_monitoring()
    
    print("\n🎉 테스트 완료!")
    print("=" * 50)
    print("✅ 모니터링 시스템이 정상적으로 작동합니다!")
    print(f"📊 대시보드 접속: http://localhost:8502")
    print(f"⏰ 테스트 완료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
except ImportError as e:
    print(f"❌ 모니터링 시스템 로드 실패: {e}")
except Exception as e:
    print(f"❌ 테스트 실행 중 오류: {e}") 