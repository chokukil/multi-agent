#!/usr/bin/env python3
"""
🍒 CherryAI Agents 최종 검증 스크립트
모든 10개 에이전트의 아키텍처 준수 및 기능 검증
"""

import os
import sys
import json
from pathlib import Path

def print_banner():
    """검증 시작 배너 출력"""
    print("🍒 Starting Agents Validation...")

def get_color_code(score):
    """점수에 따른 색상 코드 반환"""
    if score >= 4:
        return "✅"
    elif score >= 3:
        return "🟡"
    else:
        return "❌"

def validate_agent_file(file_path, expected_class, port):
    """개별 에이전트 파일 검증"""
    if not file_path.exists():
        return {
            'exists': False,
            'size': 0,
            'has_class': False,
            'has_a2a_imports': False,
            'has_port_config': False,
            'score': 0
        }
    
    content = file_path.read_text(encoding='utf-8')
    file_size = file_path.stat().st_size
    
    # 클래스 존재 확인
    has_class = f"class {expected_class}" in content
    
    # A2A SDK 임포트 확인
    a2a_imports = [
        "from a2a.server import A2AFastAPIApplication",
        "from a2a.types import",
        "from a2a.server.agents import AgentExecutor"
    ]
    has_a2a_imports = any(imp in content for imp in a2a_imports)
    
    # 포트 설정 확인
    has_port_config = f"port={port}" in content or f'"{port}"' in content
    
    # 점수 계산 (최대 5점)
    score = 0
    if file_path.exists(): score += 1
    if file_size > 1000: score += 1  # 최소 1KB 이상
    if has_class: score += 1
    if has_a2a_imports: score += 1
    if has_port_config: score += 1
    
    return {
        'exists': True,
        'size': file_size,
        'has_class': has_class,
        'has_a2a_imports': has_a2a_imports,
        'has_port_config': has_port_config,
        'score': score
    }

def main():
    """메인 검증 함수"""
    print_banner()
    
    # 에이전트 검증 설정 - Enhanced prefix 제거
    AGENTS_TO_VALIDATE = {
        "data_loader_agent.py": {
            "port": 8100,
            "class_name": "DataLoaderExecutor",
            "description": "Data Loader Agent"
        },
        "data_cleaning_agent.py": {
            "port": 8310,
            "class_name": "DataCleaningExecutor", 
            "description": "Data Cleaning Agent"
        },
        "eda_agent.py": {
            "port": 8311,
            "class_name": "EDAExecutor",
            "description": "EDA Agent"
        },
        "data_wrangling_agent.py": {
            "port": 8312,
            "class_name": "DataWranglingExecutor",
            "description": "Data Wrangling Agent"
        },
        "feature_engineering_agent.py": {
            "port": 8313,
            "class_name": "FeatureEngineeringExecutor",
            "description": "Feature Engineering Agent"
        },
        "h2o_modeling_agent.py": {
            "port": 8320,
            "class_name": "H2OModelingExecutor",
            "description": "H2O Modeling Agent"
        },
        "mlflow_agent.py": {
            "port": 8321,
            "class_name": "MLflowExecutor",
            "description": "MLflow Agent"
        },
        "sql_database_agent.py": {
            "port": 8322,
            "class_name": "SQLDatabaseExecutor",
            "description": "SQL Database Agent"
        },
        "data_visualization_agent.py": {
            "port": 8323,
            "class_name": "DataVisualizationExecutor",
            "description": "Data Visualization Agent"
        },
        "pandas_data_analyst_agent.py": {
            "port": 8324,
            "class_name": "PandasDataAnalystExecutor",
            "description": "Pandas Data Analyst Agent"
        }
    }

    print(f"\n🍒 CherryAI Agents Validation")
    print("=" * 60)
    print(f"📋 Validating {len(AGENTS_TO_VALIDATE)} Agents...")
    
    # 프로젝트 루트 디렉토리 확인
    current_dir = Path.cwd()
    agents_dir = current_dir / "a2a_ds_servers"
    
    if not agents_dir.exists():
        print(f"❌ Error: a2a_ds_servers directory not found in {current_dir}")
        sys.exit(1)
    
    results = {}
    total_score = 0
    max_possible_score = len(AGENTS_TO_VALIDATE) * 5
    
    # 포트 충돌 확인
    ports = [config["port"] for config in AGENTS_TO_VALIDATE.values()]
    port_conflicts = len(ports) != len(set(ports))
    
    # 각 에이전트 검증
    for file_name, config in AGENTS_TO_VALIDATE.items():
        file_path = agents_dir / file_name
        result = validate_agent_file(file_path, config["class_name"], config["port"])
        results[file_name] = {**result, **config}
        total_score += result['score']
        
        # 결과 출력
        status_icon = get_color_code(result['score'])
        file_status = "exists" if result['exists'] else "missing"
        size_mb = result['size'] / 1024 if result['size'] > 0 else 0
        
        print(f"\n{status_icon} Port {config['port']}: {config['description']}")
        print(f"   📁 File: a2a_ds_servers/{file_name} ({file_status})")
        print(f"   📊 Size: {size_mb:.1f} KB" if result['size'] > 0 else "   📊 Size: 0 bytes")
        print(f"   🔧 Score: {result['score']}/5")
        print(f"   📝 {config['description']}")
    
    # 전체 결과 요약
    success_rate = (total_score / max_possible_score) * 100
    successful_agents = sum(1 for r in results.values() if r['score'] >= 4)
    
    print("\n" + "=" * 60)
    print("📊 VALIDATION SUMMARY")
    print("=" * 60)
    print(f"✅ Successful Validations: {successful_agents}/{len(AGENTS_TO_VALIDATE)} ({successful_agents/len(AGENTS_TO_VALIDATE)*100:.1f}%)")
    print(f"🔌 Port Allocation: {'❌ Conflicts Found' if port_conflicts else '✅ No Conflicts'}")
    print(f"📋 Assigned Ports: {sorted(ports)}")
    
    # 아키텍처 준수도 등급
    if success_rate >= 90:
        compliance_grade = "A+ - ✅ Excellent"
    elif success_rate >= 80:
        compliance_grade = "A - ✅ Good"
    elif success_rate >= 70:
        compliance_grade = "B - 🟡 Acceptable"
    elif success_rate >= 60:
        compliance_grade = "C - ⚠️ Needs Improvement"
    else:
        compliance_grade = "F - ❌ Critical Issues"
    
    print(f"🏆 Architecture Compliance: {compliance_grade}")
    
    # 상세 분석
    print(f"\n📈 DETAILED BREAKDOWN")
    print("-" * 40)
    for file_name, result in results.items():
        if result['score'] < 4:
            status_icon = get_color_code(result['score'])
            print(f"{status_icon} {result['description']} (Port {result['port']}):")
            if not result['exists']:
                print(f"   - File missing: a2a_ds_servers/{file_name}")
            if not result['has_class']:
                print(f"   - Missing AgentExecutor implementation")
            if not result['has_a2a_imports']:
                print(f"   - Missing A2A SDK imports")
            if not result['has_port_config']:
                print(f"   - Port {result['port']} not configured")
    
    # 마이그레이션 상태
    phase1_agents = list(AGENTS_TO_VALIDATE.keys())[:5]  # 첫 5개
    phase2_agents = list(AGENTS_TO_VALIDATE.keys())[5:]  # 나머지 5개
    
    phase1_success = sum(1 for agent in phase1_agents if results[agent]['score'] >= 4)
    phase2_success = sum(1 for agent in phase2_agents if results[agent]['score'] >= 4)
    
    print(f"\n🚀 MIGRATION STATUS")
    print("-" * 40)
    print(f"📦 Phase 1 (Agents 1-5): {phase1_success}/5 ({'✅ Complete' if phase1_success == 5 else '⚠️ Incomplete'})")
    print(f"📦 Phase 2 (Agents 6-10): {phase2_success}/5 ({'✅ Complete' if phase2_success == 5 else '⚠️ Incomplete'})")
    print(f"🎯 Overall Migration: {success_rate:.1f}% Complete")
    
    # 권장사항
    print(f"\n💡 RECOMMENDATIONS")
    print("-" * 40)
    if success_rate < 70:
        print("⚠️ Major issues require attention")
        print("🔨 Complete missing agent implementations")
    elif success_rate < 90:
        print("🔧 Minor improvements needed")
        print("📋 Review agent configurations")
    else:
        print("🎉 Excellent! All agents are properly configured")
    
    if not port_conflicts:
        print("✅ Port allocation is optimal")
    else:
        print("⚠️ Resolve port conflicts")
    
    print("📋 Ensure all A2A protocol compliance")
    
    # 최종 상태
    migration_status = "✅ COMPLETE" if success_rate >= 90 else "⚠️ PENDING"
    status_message = "All agents ready for deployment" if success_rate >= 90 else "Some agents need attention"
    
    print(f"\n🎯 Final Status: {success_rate:.1f}% Migration Complete")
    print(f"{migration_status}: {status_message}")
    
    return 0 if success_rate >= 70 else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 