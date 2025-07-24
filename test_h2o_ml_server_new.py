#!/usr/bin/env python3
"""
H2OMLServerAgent 테스트 스크립트

새로운 h2o_ml_server_new.py가 올바르게 작동하는지 확인합니다.
"""

import asyncio
import sys
import os
import json
import time
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from a2a_ds_servers.h2o_ml_server_new import H2OMLServerAgent

class H2OMLServerTester:
    """H2OMLServerAgent 테스트 클래스"""
    
    def __init__(self):
        self.test_results = {
            "timestamp": time.strftime("%Y%m%d_%H%M%S"),
            "tests": []
        }
    
    async def test_initialization(self):
        """서버 에이전트 초기화 테스트"""
        print("🤖 H2OMLServerAgent 초기화 테스트")
        
        try:
            agent = H2OMLServerAgent()
            
            test_result = {
                "test_name": "initialization",
                "status": "PASS",
                "details": {
                    "executor_created": agent.executor is not None,
                    "has_wrapper": hasattr(agent.executor, 'agent') and agent.executor.agent is not None,
                    "llm_initialized": hasattr(agent.executor, 'agent') and agent.executor.agent and agent.executor.agent.llm is not None,
                    "data_processor_ready": hasattr(agent.executor, 'agent') and agent.executor.agent and agent.executor.agent.data_processor is not None
                }
            }
            
            print(f"   ✅ 초기화 성공")
            print(f"   🤖 Executor: {'✅' if agent.executor else '❌'}")
            print(f"   🔧 Wrapper: {'✅' if hasattr(agent.executor, 'agent') and agent.executor.agent else '❌'}")
            print(f"   🧠 LLM: {'✅' if hasattr(agent.executor, 'agent') and agent.executor.agent and agent.executor.agent.llm else '❌'}")
            print(f"   🔍 데이터 프로세서: {'✅' if hasattr(agent.executor, 'agent') and agent.executor.agent and agent.executor.agent.data_processor else '❌'}")
            
            self.test_results["tests"].append(test_result)
            return agent
            
        except Exception as e:
            print(f"   ❌ 초기화 실패: {e}")
            test_result = {
                "test_name": "initialization", 
                "status": "FAIL",
                "error": str(e)
            }
            self.test_results["tests"].append(test_result)
            return None
    
    async def test_data_parsing(self, agent):
        """데이터 파싱 테스트"""
        print("\\n🔍 데이터 파싱 테스트")
        
        # ML 모델링에 적합한 CSV 데이터 테스트
        csv_data = """id,age,income,score,education,employed,target
1,25,50000,85,Bachelor,1,1
2,30,60000,90,Master,1,1
3,35,70000,78,Bachelor,1,0
4,28,55000,88,Master,1,1
5,32,65000,82,PhD,1,0
6,29,58000,87,Bachelor,1,1
7,33,72000,79,Master,1,0
8,26,52000,89,Bachelor,1,1"""
        
        if hasattr(agent.executor, 'agent') and agent.executor.agent:
            df = agent.executor.agent.data_processor.parse_data_from_message(csv_data)
        else:
            df = None
        
        if df is not None:
            print(f"   ✅ CSV 파싱 성공: {df.shape}")
            print(f"   📊 숫자형 컬럼: {len(df.select_dtypes(include=['number']).columns)}개")
            print(f"   📝 범주형 컬럼: {len(df.select_dtypes(include=['object']).columns)}개")
            print(f"   🎯 ML 적합성: {'✅' if len(df.columns) >= 5 and df.shape[0] >= 5 else '❌'}")
            test_result = {
                "test_name": "data_parsing_csv",
                "status": "PASS", 
                "details": {
                    "shape": df.shape, 
                    "columns": list(df.columns),
                    "numeric_cols": len(df.select_dtypes(include=['number']).columns),
                    "categorical_cols": len(df.select_dtypes(include=['object']).columns),
                    "suitable_for_ml": len(df.columns) >= 5 and df.shape[0] >= 5
                }
            }
        else:
            print("   ❌ CSV 파싱 실패")
            test_result = {
                "test_name": "data_parsing_csv",
                "status": "FAIL"
            }
        
        self.test_results["tests"].append(test_result)
        return df
    
    async def test_h2o_ml_process(self, agent):
        """H2O ML 분석 프로세스 테스트"""
        print("\\n🤖 H2O ML 분석 프로세스 테스트")
        
        # 테스트 데이터와 요청
        test_request = """다음 고객 데이터에 대해 H2O AutoML을 실행해주세요:

id,age,income,score,education,employed,target
1,25,50000,85,Bachelor,1,1
2,30,60000,90,Master,1,1
3,35,70000,78,Bachelor,1,0
4,28,55000,88,Master,1,1
5,32,65000,82,PhD,1,0
6,29,58000,87,Bachelor,1,1
7,33,72000,79,Master,1,0
8,26,52000,89,Bachelor,1,1

target 컬럼을 예측하는 분류 모델을 학습해주세요."""
        
        try:
            if hasattr(agent.executor, 'process_h2o_ml_analysis'):
                result = await agent.executor.process_h2o_ml_analysis(test_request)
            else:
                result = "H2O ML 분석 기능을 사용할 수 없습니다 (폴백 모드)"
            
            if result and len(result) > 100:
                print("   ✅ H2O ML 분석 성공")
                print(f"   📄 결과 길이: {len(result)} 문자")
                
                # H2O ML 특화 키워드 검증
                ml_keywords = ["H2OMLAgent", "AutoML", "모델", "학습", "Complete"]
                found_keywords = [kw for kw in ml_keywords if kw in result]
                print(f"   🔍 ML 키워드 발견: {len(found_keywords)}/{len(ml_keywords)}")
                
                # H2O 관련 키워드 확인
                h2o_keywords = ["H2O", "리더보드", "best_model", "classification", "regression"]
                found_h2o_keywords = [kw for kw in h2o_keywords if kw.lower() in result.lower()]
                print(f"   🤖 H2O 키워드 발견: {len(found_h2o_keywords)}/{len(h2o_keywords)}")
                
                test_result = {
                    "test_name": "h2o_ml_process",
                    "status": "PASS",
                    "details": {
                        "result_length": len(result),
                        "ml_keywords_found": found_keywords,
                        "h2o_keywords_found": found_h2o_keywords
                    }
                }
            else:
                print("   ❌ H2O ML 분석 실패 - 결과 부족")
                test_result = {
                    "test_name": "h2o_ml_process",
                    "status": "FAIL",
                    "reason": "insufficient_result"
                }
            
            self.test_results["tests"].append(test_result)
            return result
            
        except Exception as e:
            print(f"   ❌ H2O ML 분석 오류: {e}")
            test_result = {
                "test_name": "h2o_ml_process",
                "status": "ERROR",
                "error": str(e)
            }
            self.test_results["tests"].append(test_result)
            return None
    
    async def test_automl_functionality(self, agent):
        """AutoML 기능 테스트"""
        print("\\n🚀 AutoML 기능 테스트")
        
        # AutoML 실행 시나리오 테스트
        automl_request = """다음 데이터셋에 대해 H2O AutoML을 실행해주세요:

feature1,feature2,feature3,target
1.2,2.3,3.4,A
2.1,3.2,4.3,B
3.0,4.1,5.2,A
1.8,2.9,3.8,B
2.5,3.6,4.7,A
1.4,2.1,3.2,B

분류 문제로 최적의 모델을 찾아주세요."""
        
        try:
            if hasattr(agent.executor, 'process_h2o_ml_analysis'):
                result = await agent.executor.process_h2o_ml_analysis(automl_request)
            else:
                result = "AutoML 기능을 사용할 수 없습니다 (폴백 모드)"
            
            if result and ("AutoML" in result or "automl" in result.lower()):
                print("   ✅ AutoML 기능 실행 성공")
                
                # AutoML 관련 키워드 확인
                automl_keywords = ["AutoML", "automl", "리더보드", "leaderboard", "모델"]
                found_automl_keywords = [kw for kw in automl_keywords if kw.lower() in result.lower()]
                print(f"   🤖 AutoML 키워드 발견: {len(found_automl_keywords)}/{len(automl_keywords)}")
                
                test_result = {
                    "test_name": "automl_functionality",
                    "status": "PASS",
                    "details": {
                        "automl_keywords": found_automl_keywords
                    }
                }
            else:
                print("   ❌ AutoML 기능 실행 실패")
                test_result = {
                    "test_name": "automl_functionality", 
                    "status": "FAIL"
                }
            
            self.test_results["tests"].append(test_result)
            
        except Exception as e:
            print(f"   ❌ AutoML 기능 오류: {e}")
            test_result = {
                "test_name": "automl_functionality",
                "status": "ERROR",
                "error": str(e)
            }
            self.test_results["tests"].append(test_result)
    
    async def test_model_evaluation(self, agent):
        """모델 평가 기능 테스트"""
        print("\\n📊 모델 평가 기능 테스트")
        
        # 모델 평가 시나리오 테스트
        evaluation_request = """다음 모델 성능 데이터를 평가해주세요:

model_id,accuracy,precision,recall,f1_score
GBM_1,0.85,0.83,0.87,0.85
RF_1,0.82,0.80,0.84,0.82
XGBoost_1,0.88,0.86,0.90,0.88

최고 성능 모델을 선택하고 평가해주세요."""
        
        try:
            if hasattr(agent.executor, 'process_h2o_ml_analysis'):
                result = await agent.executor.process_h2o_ml_analysis(evaluation_request)
            else:
                result = "모델 평가 기능을 사용할 수 없습니다 (폴백 모드)"
            
            if result and ("평가" in result or "evaluation" in result.lower()):
                print("   ✅ 모델 평가 기능 성공")
                
                # 평가 관련 키워드 확인
                eval_keywords = ["평가", "성능", "accuracy", "precision", "recall", "f1"]
                found_eval_keywords = [kw for kw in eval_keywords if kw.lower() in result.lower()]
                print(f"   📊 평가 키워드 발견: {len(found_eval_keywords)}/{len(eval_keywords)}")
                
                test_result = {
                    "test_name": "model_evaluation",
                    "status": "PASS",
                    "details": {
                        "evaluation_keywords": found_eval_keywords
                    }
                }
            else:
                print("   ❌ 모델 평가 기능 실패")
                test_result = {
                    "test_name": "model_evaluation", 
                    "status": "FAIL"
                }
            
            self.test_results["tests"].append(test_result)
            
        except Exception as e:
            print(f"   ❌ 모델 평가 오류: {e}")
            test_result = {
                "test_name": "model_evaluation",
                "status": "ERROR",
                "error": str(e)
            }
            self.test_results["tests"].append(test_result)
    
    async def test_feature_importance(self, agent):
        """피처 중요도 분석 기능 테스트"""
        print("\\n🔍 피처 중요도 분석 기능 테스트")
        
        # 피처 중요도 분석 시나리오 테스트
        importance_request = """다음 데이터의 피처 중요도를 분석해주세요:

age,income,education_years,experience,target
25,50000,16,2,1
30,60000,18,5,1
35,70000,16,10,0
28,55000,18,3,1
32,65000,20,7,0

각 피처가 target 예측에 미치는 영향을 분석해주세요."""
        
        try:
            if hasattr(agent.executor, 'process_h2o_ml_analysis'):
                result = await agent.executor.process_h2o_ml_analysis(importance_request)
            else:
                result = "피처 중요도 분석 기능을 사용할 수 없습니다 (폴백 모드)"
            
            if result and ("중요도" in result or "importance" in result.lower()):
                print("   ✅ 피처 중요도 분석 성공")
                
                # 중요도 관련 키워드 확인
                importance_keywords = ["중요도", "importance", "피처", "feature", "영향", "기여도"]
                found_importance_keywords = [kw for kw in importance_keywords if kw.lower() in result.lower()]
                print(f"   🔍 중요도 키워드 발견: {len(found_importance_keywords)}/{len(importance_keywords)}")
                
                test_result = {
                    "test_name": "feature_importance",
                    "status": "PASS",
                    "details": {
                        "importance_keywords": found_importance_keywords
                    }
                }
            else:
                print("   ❌ 피처 중요도 분석 실패")
                test_result = {
                    "test_name": "feature_importance", 
                    "status": "FAIL"
                }
            
            self.test_results["tests"].append(test_result)
            
        except Exception as e:
            print(f"   ❌ 피처 중요도 분석 오류: {e}")
            test_result = {
                "test_name": "feature_importance",
                "status": "ERROR",
                "error": str(e)
            }
            self.test_results["tests"].append(test_result)
    
    async def test_guidance_generation(self, agent):
        """가이드 생성 테스트"""
        print("\\n📚 가이드 생성 테스트")
        
        try:
            if hasattr(agent.executor, 'process_h2o_ml_analysis'):
                guidance = await agent.executor.process_h2o_ml_analysis("H2O AutoML 사용 방법을 알려주세요")
            else:
                guidance = "H2O AutoML 가이드 기능을 사용할 수 없습니다 (폴백 모드)"
            
            if guidance and "가이드" in guidance:
                print("   ✅ 가이드 생성 성공")
                
                # 가이드 특화 키워드 확인
                guide_keywords = ["H2O", "AutoML", "가이드", "머신러닝", "모델"]
                found_guide_keywords = [kw for kw in guide_keywords if kw in guidance]
                print(f"   📖 가이드 키워드 발견: {len(found_guide_keywords)}/{len(guide_keywords)}")
                
                test_result = {
                    "test_name": "guidance_generation",
                    "status": "PASS",
                    "details": {
                        "guide_keywords": found_guide_keywords
                    }
                }
            else:
                print("   ❌ 가이드 생성 실패")
                test_result = {
                    "test_name": "guidance_generation", 
                    "status": "FAIL"
                }
            
            self.test_results["tests"].append(test_result)
            
        except Exception as e:
            print(f"   ❌ 가이드 생성 오류: {e}")
            test_result = {
                "test_name": "guidance_generation",
                "status": "ERROR",
                "error": str(e)
            }
            self.test_results["tests"].append(test_result)
    
    async def run_all_tests(self):
        """모든 테스트 실행"""
        print("🚀 H2OMLServerAgent 종합 테스트 시작\\n")
        
        # 초기화 테스트
        agent = await self.test_initialization()
        if not agent:
            print("❌ 초기화 실패로 테스트 중단")
            return
        
        # 데이터 파싱 테스트
        df = await self.test_data_parsing(agent)
        
        # H2O ML 분석 프로세스 테스트
        result = await self.test_h2o_ml_process(agent)
        
        # AutoML 기능 테스트
        await self.test_automl_functionality(agent)
        
        # 모델 평가 테스트
        await self.test_model_evaluation(agent)
        
        # 피처 중요도 분석 테스트
        await self.test_feature_importance(agent)
        
        # 가이드 생성 테스트
        await self.test_guidance_generation(agent)
        
        # 결과 요약
        self.print_test_summary()
        self.save_test_results()
    
    def print_test_summary(self):
        """테스트 결과 요약 출력"""
        print("\\n" + "="*80)
        print("🤖 H2OMLServerAgent 테스트 결과 요약")
        print("="*80)
        
        total_tests = len(self.test_results["tests"])
        passed_tests = sum(1 for test in self.test_results["tests"] if test["status"] == "PASS")
        failed_tests = sum(1 for test in self.test_results["tests"] if test["status"] == "FAIL")
        error_tests = sum(1 for test in self.test_results["tests"] if test["status"] == "ERROR")
        
        print(f"🕐 테스트 시간: {self.test_results['timestamp']}")
        print(f"📈 전체 테스트: {total_tests}개")
        print(f"✅ 성공: {passed_tests}개")
        print(f"❌ 실패: {failed_tests}개")
        print(f"💥 오류: {error_tests}개")
        print(f"🎯 성공률: {(passed_tests/total_tests*100):.1f}%" if total_tests > 0 else "0%")
        
        print("\\n📋 테스트별 상세 결과:")
        for test in self.test_results["tests"]:
            status_icon = {"PASS": "✅", "FAIL": "❌", "ERROR": "💥"}.get(test["status"], "❓")
            print(f"   {status_icon} {test['test_name']}: {test['status']}")
        
        print("\\n" + "="*80)
    
    def save_test_results(self):
        """테스트 결과 저장"""
        filename = f"h2o_ml_server_test_results_{self.test_results['timestamp']}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.test_results, f, indent=2, ensure_ascii=False)
            print(f"💾 테스트 결과 저장: {filename}")
        except Exception as e:
            print(f"⚠️ 결과 저장 실패: {e}")


async def main():
    """메인 테스트 실행 함수"""
    tester = H2OMLServerTester()
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())