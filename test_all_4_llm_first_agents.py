#!/usr/bin/env python3
"""
All 4 LLM-First Agents Comprehensive Test
테스트 대상: SQL Database, MLflow Tools, Pandas Analyst, Report Generator
"""

import subprocess
import time
import requests
import json
import logging
import sys
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 에이전트 서버 설정
AGENTS = {
    "SQL Database": {
        "server_file": "a2a_ds_servers/sql_database_server_new.py",
        "port": 8311,
        "test_message": "CREATE TABLE test_table (id INTEGER, name TEXT); INSERT INTO test_table VALUES (1, 'Alice'), (2, 'Bob'); SELECT * FROM test_table;"
    },
    "MLflow Tools": {
        "server_file": "a2a_ds_servers/mlflow_tools_server_new.py", 
        "port": 8314,
        "test_message": "Create a new MLflow experiment called 'test_experiment' and start a run to track model performance"
    },
    "Pandas Analyst": {
        "server_file": "a2a_ds_servers/pandas_analyst_server_new.py",
        "port": 8315,
        "test_message": "name,age,city\nAlice,25,Seoul\nBob,30,Busan\nCharlie,35,Daegu\n\nAnalyze this data and show basic statistics"
    },
    "Report Generator": {
        "server_file": "a2a_ds_servers/report_generator_server_new.py",
        "port": 8316,
        "test_message": "quarter,revenue,profit\nQ1,100000,15000\nQ2,120000,18000\nQ3,110000,16500\nQ4,130000,20000\n\nGenerate a comprehensive business report"
    }
}

def start_server(agent_name: str, server_file: str, port: int) -> subprocess.Popen:
    """서버 시작"""
    try:
        logger.info(f"🚀 Starting {agent_name} server on port {port}")
        process = subprocess.Popen(
            [sys.executable, server_file],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # 서버 시작 대기
        time.sleep(5)
        
        # 서버 상태 확인 (A2A health check - process is running if we get here)
        if process.poll() is None:  # Process is still running
            logger.info(f"✅ {agent_name} server started successfully on port {port}")
            return process
        else:
            logger.error(f"❌ {agent_name} server process died")
            return None
            
    except Exception as e:
        logger.error(f"❌ Failed to start {agent_name} server: {e}")
        return None

def test_agent(agent_name: str, port: int, test_message: str) -> dict:
    """에이전트 테스트"""
    try:
        logger.info(f"🧪 Testing {agent_name}...")
        
        # A2A JSON-RPC 2.0 요청 데이터 구성
        message_id = f"test_{agent_name.lower().replace(' ', '_')}_{int(time.time())}"
        request_data = {
            "jsonrpc": "2.0",
            "method": "message/send",
            "params": {
                "message": {
                    "messageId": message_id,
                    "role": "user",
                    "parts": [
                        {
                            "kind": "text",
                            "text": test_message
                        }
                    ]
                }
            },
            "id": message_id
        }
        
        # 요청 전송 (root endpoint)
        response = requests.post(
            f"http://localhost:{port}/",
            json=request_data,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"✅ {agent_name} test successful")
            return {
                "agent": agent_name,
                "port": port,
                "status": "success",
                "response_length": len(str(result)),
                "has_task_id": "taskId" in result,
                "response_preview": str(result)[:200] + "..." if len(str(result)) > 200 else str(result)
            }
        else:
            logger.error(f"❌ {agent_name} test failed with status {response.status_code}")
            return {
                "agent": agent_name,
                "port": port,
                "status": "failed",
                "error": f"HTTP {response.status_code}: {response.text}"
            }
            
    except Exception as e:
        logger.error(f"❌ {agent_name} test error: {e}")
        return {
            "agent": agent_name,
            "port": port,
            "status": "error",
            "error": str(e)
        }

def stop_server(process: subprocess.Popen, agent_name: str):
    """서버 중지"""
    if process:
        try:
            process.terminate()
            process.wait(timeout=5)
            logger.info(f"🛑 {agent_name} server stopped")
        except subprocess.TimeoutExpired:
            process.kill()
            logger.warning(f"⚠️ {agent_name} server forcefully killed")
        except Exception as e:
            logger.error(f"❌ Error stopping {agent_name} server: {e}")

def main():
    """메인 테스트 함수"""
    logger.info("🔥 Starting comprehensive test of all 4 LLM-First agents")
    logger.info("=" * 80)
    
    # 테스트 결과
    test_results = []
    running_processes = {}
    
    try:
        # 1단계: 모든 서버 시작
        logger.info("📡 Phase 1: Starting all agent servers")
        for agent_name, config in AGENTS.items():
            process = start_server(agent_name, config["server_file"], config["port"])
            if process:
                running_processes[agent_name] = process
            else:
                logger.error(f"❌ Failed to start {agent_name} - skipping test")
        
        logger.info(f"✅ Started {len(running_processes)}/{len(AGENTS)} servers successfully")
        
        # 2단계: 모든 에이전트 테스트
        logger.info("\n🧪 Phase 2: Testing all agents")
        for agent_name, config in AGENTS.items():
            if agent_name in running_processes:
                result = test_agent(agent_name, config["port"], config["test_message"])
                test_results.append(result)
            else:
                test_results.append({
                    "agent": agent_name,
                    "port": config["port"],
                    "status": "server_failed",
                    "error": "Server failed to start"
                })
        
        # 3단계: 결과 분석
        logger.info("\n📊 Phase 3: Test Results Analysis")
        logger.info("=" * 80)
        
        successful_tests = sum(1 for r in test_results if r["status"] == "success")
        total_tests = len(test_results)
        success_rate = (successful_tests / total_tests) * 100
        
        logger.info(f"🎯 Overall Success Rate: {successful_tests}/{total_tests} ({success_rate:.1f}%)")
        
        # 개별 결과 출력
        for result in test_results:
            agent = result["agent"]
            status = result["status"]
            
            if status == "success":
                logger.info(f"✅ {agent}: SUCCESS (Port {result['port']})")
                logger.info(f"   - Response length: {result['response_length']} chars")
                logger.info(f"   - Has task ID: {result['has_task_id']}")
            else:
                logger.error(f"❌ {agent}: {status.upper()} (Port {result['port']})")
                if "error" in result:
                    logger.error(f"   - Error: {result['error']}")
        
        # 결과 저장
        timestamp = int(time.time())
        result_file = f"llm_first_agents_test_results_{timestamp}.json"
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump({
                "timestamp": timestamp,
                "total_agents": total_tests,
                "successful_tests": successful_tests,
                "success_rate": success_rate,
                "detailed_results": test_results
            }, f, ensure_ascii=False, indent=2)
        
        logger.info(f"📁 Test results saved to: {result_file}")
        
        # 최종 결과
        if success_rate == 100:
            logger.info("🎉 ALL LLM-FIRST AGENTS WORKING PERFECTLY!")
        elif success_rate >= 75:
            logger.info("🔥 MOST AGENTS WORKING - MINOR ISSUES DETECTED")
        else:
            logger.warning("⚠️ SIGNIFICANT ISSUES DETECTED - REQUIRES ATTENTION")
            
    except KeyboardInterrupt:
        logger.info("🛑 Test interrupted by user")
        
    finally:
        # 4단계: 서버 정리
        logger.info("\n🧹 Phase 4: Cleaning up servers")
        for agent_name, process in running_processes.items():
            stop_server(process, agent_name)
        
        logger.info("✅ All servers stopped")
        logger.info("🏁 Test completed!")

if __name__ == "__main__":
    main()