#!/usr/bin/env python3
"""
FeatureEngineeringServerAgent 테스트 스크립트

새로운 feature_engineering_server_new.py가 올바르게 작동하는지 확인합니다.
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

from a2a_ds_servers.feature_engineering_server_new import FeatureEngineeringServerAgent

class FeatureEngineeringServerTester:
    """FeatureEngineeringServerAgent 테스트 클래스"""
    
    def __init__(self):
        self.test_results = {
            "timestamp": time.strftime("%Y%m%d_%H%M%S"),
            "tests": []
        }
    
    async def test_initialization(self):
        """서버 에이전트 초기화 테스트"""
        print("🔧 FeatureEngineeringServerAgent 초기화 테스트")
        
        try:
            agent = FeatureEngineeringServerAgent()
            
            test_result = {
                "test_name": "initialization",
                "status": "PASS",
                "details": {
                    "llm_initialized": agent.llm is not None,
                    "has_original_agent": agent.has_original_agent,
                    "data_processor_ready": agent.data_processor is not None
                }
            }
            
            print(f"   ✅ 초기화 성공")
            print(f"   📊 LLM: {'✅' if agent.llm else '❌'}")
            print(f"   🤖 원본 에이전트: {'✅' if agent.has_original_agent else '❌'}")
            print(f"   🔍 데이터 프로세서: {'✅' if agent.data_processor else '❌'}")
            
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
        
        # 피처 엔지니어링에 적합한 CSV 데이터 테스트
        csv_data = """id,age,category,salary,is_married,target
1,25,A,50000,true,1
2,30,B,60000,false,0
3,35,A,70000,true,1
4,28,C,55000,false,0
5,32,B,65000,true,1"""
        
        df = agent.data_processor.parse_data_from_message(csv_data)
        
        if df is not None:
            print(f"   ✅ CSV 파싱 성공: {df.shape}")
            print(f"   📊 숫자형 컬럼: {len(df.select_dtypes(include=['number']).columns)}개")
            print(f"   📝 범주형 컬럼: {len(df.select_dtypes(include=['object']).columns)}개")
            print(f"   🔢 불린형 컬럼: {len(df.select_dtypes(include=['bool']).columns)}개")
            test_result = {
                "test_name": "data_parsing_csv",
                "status": "PASS", 
                "details": {
                    "shape": df.shape, 
                    "columns": list(df.columns),
                    "numeric_cols": len(df.select_dtypes(include=['number']).columns),
                    "categorical_cols": len(df.select_dtypes(include=['object']).columns),
                    "bool_cols": len(df.select_dtypes(include=['bool']).columns)
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
    
    async def test_feature_engineering_process(self, agent):
        """피처 엔지니어링 프로세스 테스트"""
        print("\\n🔧 피처 엔지니어링 프로세스 테스트")
        
        # 테스트 데이터와 요청
        test_request = """다음 데이터에 대해 피처 엔지니어링을 수행해주세요:

id,age,category,salary,is_married,target
1,25,A,50000,true,1
2,30,B,60000,false,0
3,35,A,70000,true,1
4,28,C,55000,false,0
5,32,B,65000,true,1
6,29,A,58000,false,0

범주형 변수를 인코딩하고 불린 변수를 정수로 변환해주세요."""
        
        try:
            result = await agent.process_feature_engineering(test_request)
            
            if result and len(result) > 100:
                print("   ✅ 피처 엔지니어링 성공")
                print(f"   📄 결과 길이: {len(result)} 문자")
                
                # 피처 엔지니어링 특화 키워드 검증
                engineering_keywords = ["FeatureEngineeringAgent", "피처", "엔지니어링", "인코딩", "Complete"]
                found_keywords = [kw for kw in engineering_keywords if kw in result]
                print(f"   🔍 엔지니어링 키워드 발견: {len(found_keywords)}/{len(engineering_keywords)}")
                
                # 변환 관련 키워드 확인
                transform_keywords = ["encode", "인코딩", "변환", "category", "bool"]
                found_transform_keywords = [kw for kw in transform_keywords if kw.lower() in result.lower()]
                print(f"   🔄 변환 키워드 발견: {len(found_transform_keywords)}/{len(transform_keywords)}")
                
                test_result = {
                    "test_name": "feature_engineering_process",
                    "status": "PASS",
                    "details": {
                        "result_length": len(result),
                        "engineering_keywords_found": found_keywords,
                        "transform_keywords_found": found_transform_keywords
                    }
                }
            else:
                print("   ❌ 피처 엔지니어링 실패 - 결과 부족")
                test_result = {
                    "test_name": "feature_engineering_process",
                    "status": "FAIL",
                    "reason": "insufficient_result"
                }
            
            self.test_results["tests"].append(test_result)
            return result
            
        except Exception as e:
            print(f"   ❌ 피처 엔지니어링 오류: {e}")
            test_result = {
                "test_name": "feature_engineering_process",
                "status": "ERROR",
                "error": str(e)
            }
            self.test_results["tests"].append(test_result)
            return None
    
    async def test_categorical_encoding(self, agent):
        """범주형 인코딩 기능 테스트"""
        print("\\n🏷️ 범주형 인코딩 기능 테스트")
        
        # 인코딩 시나리오 테스트
        encoding_request = """다음 데이터의 범주형 변수들을 원핫 인코딩해주세요:

product_id,category,brand,price,rating
1,Electronics,Samsung,1000,4.5
2,Clothing,Nike,80,4.2
3,Electronics,Apple,1200,4.8
4,Books,Penguin,15,4.0
5,Clothing,Adidas,90,4.3"""
        
        try:
            result = await agent.process_feature_engineering(encoding_request)
            
            if result and ("인코딩" in result or "encoding" in result.lower()):
                print("   ✅ 범주형 인코딩 성공")
                
                # 인코딩 관련 키워드 확인
                encoding_keywords = ["원핫", "onehot", "encoding", "범주형", "category"]
                found_encoding_keywords = [kw for kw in encoding_keywords if kw.lower() in result.lower()]
                print(f"   🏷️ 인코딩 키워드 발견: {len(found_encoding_keywords)}/{len(encoding_keywords)}")
                
                test_result = {
                    "test_name": "categorical_encoding",
                    "status": "PASS",
                    "details": {
                        "encoding_keywords": found_encoding_keywords
                    }
                }
            else:
                print("   ❌ 범주형 인코딩 실패")
                test_result = {
                    "test_name": "categorical_encoding", 
                    "status": "FAIL"
                }
            
            self.test_results["tests"].append(test_result)
            
        except Exception as e:
            print(f"   ❌ 범주형 인코딩 오류: {e}")
            test_result = {
                "test_name": "categorical_encoding",
                "status": "ERROR",
                "error": str(e)
            }
            self.test_results["tests"].append(test_result)
    
    async def test_target_detection(self, agent):
        """타겟 변수 감지 테스트"""
        print("\\n🎯 타겟 변수 감지 테스트")
        
        # 다양한 데이터셋으로 타겟 감지 테스트
        target_datasets = [
            {
                "name": "명시적 target 컬럼",
                "data": """feature1,feature2,target
1,2,1
3,4,0
5,6,1""",
                "expected_target": "target"
            },
            {
                "name": "churn 컬럼",
                "data": """customer_id,age,churn
1,25,1
2,30,0
3,35,1""",
                "expected_target": "churn"
            }
        ]
        
        for dataset in target_datasets:
            try:
                df = agent.data_processor.parse_data_from_message(dataset["data"])
                if df is not None:
                    detected_target = agent._detect_target_variable(df, dataset["data"])
                    
                    if detected_target == dataset["expected_target"]:
                        print(f"   ✅ {dataset['name']}: {detected_target}")
                        test_result = {
                            "test_name": f"target_detection_{dataset['name']}",
                            "status": "PASS",
                            "details": {
                                "detected_target": detected_target,
                                "expected_target": dataset["expected_target"]
                            }
                        }
                    else:
                        print(f"   ❌ {dataset['name']}: 예상 {dataset['expected_target']}, 감지 {detected_target}")
                        test_result = {
                            "test_name": f"target_detection_{dataset['name']}",
                            "status": "FAIL",
                            "details": {
                                "detected_target": detected_target,
                                "expected_target": dataset["expected_target"]
                            }
                        }
                else:
                    print(f"   ❌ {dataset['name']}: 데이터 파싱 실패")
                    test_result = {
                        "test_name": f"target_detection_{dataset['name']}",
                        "status": "FAIL",
                        "reason": "data_parsing_failed"
                    }
                
                self.test_results["tests"].append(test_result)
                
            except Exception as e:
                print(f"   ❌ {dataset['name']} 타겟 감지 오류: {e}")
                test_result = {
                    "test_name": f"target_detection_{dataset['name']}",
                    "status": "ERROR",
                    "error": str(e)
                }
                self.test_results["tests"].append(test_result)
    
    async def test_guidance_generation(self, agent):
        """가이드 생성 테스트"""
        print("\\n📚 가이드 생성 테스트")
        
        try:
            guidance = await agent.process_feature_engineering("피처 엔지니어링 방법을 알려주세요")
            
            if guidance and "가이드" in guidance:
                print("   ✅ 가이드 생성 성공")
                
                # 가이드 특화 키워드 확인
                guide_keywords = ["피처", "인코딩", "스케일링", "변환", "엔지니어링"]
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
        print("🚀 FeatureEngineeringServerAgent 종합 테스트 시작\\n")
        
        # 초기화 테스트
        agent = await self.test_initialization()
        if not agent:
            print("❌ 초기화 실패로 테스트 중단")
            return
        
        # 데이터 파싱 테스트
        df = await self.test_data_parsing(agent)
        
        # 피처 엔지니어링 프로세스 테스트
        result = await self.test_feature_engineering_process(agent)
        
        # 범주형 인코딩 테스트
        await self.test_categorical_encoding(agent)
        
        # 타겟 변수 감지 테스트
        await self.test_target_detection(agent)
        
        # 가이드 생성 테스트
        await self.test_guidance_generation(agent)
        
        # 결과 요약
        self.print_test_summary()
        self.save_test_results()
    
    def print_test_summary(self):
        """테스트 결과 요약 출력"""
        print("\\n" + "="*80)
        print("🔧 FeatureEngineeringServerAgent 테스트 결과 요약")
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
        filename = f"feature_engineering_server_test_results_{self.test_results['timestamp']}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.test_results, f, indent=2, ensure_ascii=False)
            print(f"💾 테스트 결과 저장: {filename}")
        except Exception as e:
            print(f"⚠️ 결과 저장 실패: {e}")


async def main():
    """메인 테스트 실행 함수"""
    tester = FeatureEngineeringServerTester()
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())