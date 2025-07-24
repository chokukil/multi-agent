#!/usr/bin/env python3
"""
EDAToolsServerAgent 테스트 스크립트

새로운 eda_tools_server_new.py가 올바르게 작동하는지 확인합니다.
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

from a2a_ds_servers.eda_tools_server_new import EDAToolsServerAgent

class EDAToolsServerTester:
    """EDAToolsServerAgent 테스트 클래스"""
    
    def __init__(self):
        self.test_results = {
            "timestamp": time.strftime("%Y%m%d_%H%M%S"),
            "tests": []
        }
    
    async def test_initialization(self):
        """서버 에이전트 초기화 테스트"""
        print("📊 EDAToolsServerAgent 초기화 테스트")
        
        try:
            agent = EDAToolsServerAgent()
            
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
        
        # EDA 분석에 적합한 CSV 데이터 테스트
        csv_data = """id,age,salary,department,experience,rating
1,25,50000,IT,2,4.2
2,30,60000,HR,5,4.5
3,35,70000,Finance,8,4.1
4,28,55000,IT,3,4.3
5,32,65000,Marketing,6,4.4
6,29,58000,HR,4,4.0
7,33,72000,Finance,7,4.6
8,26,52000,IT,2,4.1"""
        
        df = agent.data_processor.parse_data_from_message(csv_data)
        
        if df is not None:
            print(f"   ✅ CSV 파싱 성공: {df.shape}")
            print(f"   📊 숫자형 컬럼: {len(df.select_dtypes(include=['number']).columns)}개")
            print(f"   📝 범주형 컬럼: {len(df.select_dtypes(include=['object']).columns)}개")
            print(f"   📈 통계 적합성: {'✅' if len(df.select_dtypes(include=['number']).columns) >= 3 else '❌'}")
            test_result = {
                "test_name": "data_parsing_csv",
                "status": "PASS", 
                "details": {
                    "shape": df.shape, 
                    "columns": list(df.columns),
                    "numeric_cols": len(df.select_dtypes(include=['number']).columns),
                    "categorical_cols": len(df.select_dtypes(include=['object']).columns),
                    "suitable_for_eda": len(df.select_dtypes(include=['number']).columns) >= 3
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
    
    async def test_eda_analysis_process(self, agent):
        """EDA 분석 프로세스 테스트"""
        print("\\n📊 EDA 분석 프로세스 테스트")
        
        # 테스트 데이터와 요청
        test_request = """다음 직원 데이터에 대해 탐색적 데이터 분석을 수행해주세요:

id,age,salary,department,experience,rating
1,25,50000,IT,2,4.2
2,30,60000,HR,5,4.5
3,35,70000,Finance,8,4.1
4,28,55000,IT,3,4.3
5,32,65000,Marketing,6,4.4
6,29,58000,HR,4,4.0
7,33,72000,Finance,7,4.6
8,26,52000,IT,2,4.1

기술 통계와 상관관계를 분석해주세요."""
        
        try:
            result = await agent.process_eda_analysis(test_request)
            
            if result and len(result) > 100:
                print("   ✅ EDA 분석 성공")
                print(f"   📄 결과 길이: {len(result)} 문자")
                
                # EDA 특화 키워드 검증
                eda_keywords = ["EDAToolsAgent", "분석", "통계", "상관관계", "Complete"]
                found_keywords = [kw for kw in eda_keywords if kw in result]
                print(f"   🔍 EDA 키워드 발견: {len(found_keywords)}/{len(eda_keywords)}")
                
                # 통계 관련 키워드 확인
                stats_keywords = ["평균", "표준편차", "상관", "correlation", "분포"]
                found_stats_keywords = [kw for kw in stats_keywords if kw.lower() in result.lower()]
                print(f"   📈 통계 키워드 발견: {len(found_stats_keywords)}/{len(stats_keywords)}")
                
                test_result = {
                    "test_name": "eda_analysis_process",
                    "status": "PASS",
                    "details": {
                        "result_length": len(result),
                        "eda_keywords_found": found_keywords,
                        "stats_keywords_found": found_stats_keywords
                    }
                }
            else:
                print("   ❌ EDA 분석 실패 - 결과 부족")
                test_result = {
                    "test_name": "eda_analysis_process",
                    "status": "FAIL",
                    "reason": "insufficient_result"
                }
            
            self.test_results["tests"].append(test_result)
            return result
            
        except Exception as e:
            print(f"   ❌ EDA 분석 오류: {e}")
            test_result = {
                "test_name": "eda_analysis_process",
                "status": "ERROR",
                "error": str(e)
            }
            self.test_results["tests"].append(test_result)
            return None
    
    async def test_descriptive_statistics(self, agent):
        """기술 통계 계산 기능 테스트"""
        print("\\n📈 기술 통계 계산 기능 테스트")
        
        # 통계 분석 시나리오 테스트
        stats_request = """다음 데이터의 기술 통계를 계산해주세요:

product_id,price,sales,rating,reviews
1,100,50,4.2,120
2,150,75,4.5,89
3,200,60,4.1,156
4,80,90,4.3,203
5,120,85,4.4,145
6,175,55,4.0,98"""
        
        try:
            result = await agent.process_eda_analysis(stats_request)
            
            if result and ("통계" in result or "평균" in result or "statistics" in result.lower()):
                print("   ✅ 기술 통계 계산 성공")
                
                # 통계 관련 키워드 확인
                stats_keywords = ["평균", "mean", "표준편차", "std", "중앙값", "median"]
                found_stats_keywords = [kw for kw in stats_keywords if kw.lower() in result.lower()]
                print(f"   📊 통계 지표 발견: {len(found_stats_keywords)}/{len(stats_keywords)}")
                
                test_result = {
                    "test_name": "descriptive_statistics",
                    "status": "PASS",
                    "details": {
                        "stats_keywords": found_stats_keywords
                    }
                }
            else:
                print("   ❌ 기술 통계 계산 실패")
                test_result = {
                    "test_name": "descriptive_statistics", 
                    "status": "FAIL"
                }
            
            self.test_results["tests"].append(test_result)
            
        except Exception as e:
            print(f"   ❌ 기술 통계 계산 오류: {e}")
            test_result = {
                "test_name": "descriptive_statistics",
                "status": "ERROR",
                "error": str(e)
            }
            self.test_results["tests"].append(test_result)
    
    async def test_correlation_analysis(self, agent):
        """상관관계 분석 기능 테스트"""
        print("\\n🔗 상관관계 분석 기능 테스트")
        
        # 상관관계 분석 시나리오 테스트
        correlation_request = """다음 데이터의 변수 간 상관관계를 분석해주세요:

student_id,math_score,science_score,study_hours,sleep_hours
1,85,88,6,7
2,92,95,8,8
3,78,82,5,6
4,88,90,7,7
5,95,98,9,8
6,82,85,6,7
7,90,93,8,8"""
        
        try:
            result = await agent.process_eda_analysis(correlation_request)
            
            if result and ("상관" in result or "correlation" in result.lower()):
                print("   ✅ 상관관계 분석 성공")
                
                # 상관관계 관련 키워드 확인
                corr_keywords = ["상관", "correlation", "pearson", "spearman", "관계"]
                found_corr_keywords = [kw for kw in corr_keywords if kw.lower() in result.lower()]
                print(f"   🔗 상관관계 키워드 발견: {len(found_corr_keywords)}/{len(corr_keywords)}")
                
                test_result = {
                    "test_name": "correlation_analysis",
                    "status": "PASS",
                    "details": {
                        "correlation_keywords": found_corr_keywords
                    }
                }
            else:
                print("   ❌ 상관관계 분석 실패")
                test_result = {
                    "test_name": "correlation_analysis", 
                    "status": "FAIL"
                }
            
            self.test_results["tests"].append(test_result)
            
        except Exception as e:
            print(f"   ❌ 상관관계 분석 오류: {e}")
            test_result = {
                "test_name": "correlation_analysis",
                "status": "ERROR",
                "error": str(e)
            }
            self.test_results["tests"].append(test_result)
    
    async def test_data_quality_assessment(self, agent):
        """데이터 품질 평가 기능 테스트"""
        print("\\n✅ 데이터 품질 평가 기능 테스트")
        
        # 품질 문제가 있는 데이터로 테스트
        quality_request = """다음 데이터의 품질을 평가해주세요:

customer_id,name,age,income,
1,Alice,25,50000,Premium
2,Bob,,60000,Standard
1,Alice,25,50000,Premium
3,Charlie,35,,Premium
4,David,30,70000,Standard
5,,28,55000,Standard"""
        
        try:
            result = await agent.process_eda_analysis(quality_request)
            
            if result and ("품질" in result or "quality" in result.lower() or "결측" in result):
                print("   ✅ 데이터 품질 평가 성공")
                
                # 품질 관련 키워드 확인
                quality_keywords = ["품질", "quality", "결측", "missing", "중복", "duplicate"]
                found_quality_keywords = [kw for kw in quality_keywords if kw.lower() in result.lower()]
                print(f"   ✅ 품질 키워드 발견: {len(found_quality_keywords)}/{len(quality_keywords)}")
                
                test_result = {
                    "test_name": "data_quality_assessment",
                    "status": "PASS",
                    "details": {
                        "quality_keywords": found_quality_keywords
                    }
                }
            else:
                print("   ❌ 데이터 품질 평가 실패")
                test_result = {
                    "test_name": "data_quality_assessment", 
                    "status": "FAIL"
                }
            
            self.test_results["tests"].append(test_result)
            
        except Exception as e:
            print(f"   ❌ 데이터 품질 평가 오류: {e}")
            test_result = {
                "test_name": "data_quality_assessment",
                "status": "ERROR",
                "error": str(e)
            }
            self.test_results["tests"].append(test_result)
    
    async def test_guidance_generation(self, agent):
        """가이드 생성 테스트"""
        print("\\n📚 가이드 생성 테스트")
        
        try:
            guidance = await agent.process_eda_analysis("EDA 분석 방법을 알려주세요")
            
            if guidance and "가이드" in guidance:
                print("   ✅ 가이드 생성 성공")
                
                # 가이드 특화 키워드 확인
                guide_keywords = ["EDA", "분석", "통계", "상관관계", "가이드"]
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
        print("🚀 EDAToolsServerAgent 종합 테스트 시작\\n")
        
        # 초기화 테스트
        agent = await self.test_initialization()
        if not agent:
            print("❌ 초기화 실패로 테스트 중단")
            return
        
        # 데이터 파싱 테스트
        df = await self.test_data_parsing(agent)
        
        # EDA 분석 프로세스 테스트
        result = await self.test_eda_analysis_process(agent)
        
        # 기술 통계 테스트
        await self.test_descriptive_statistics(agent)
        
        # 상관관계 분석 테스트
        await self.test_correlation_analysis(agent)
        
        # 데이터 품질 평가 테스트
        await self.test_data_quality_assessment(agent)
        
        # 가이드 생성 테스트
        await self.test_guidance_generation(agent)
        
        # 결과 요약
        self.print_test_summary()
        self.save_test_results()
    
    def print_test_summary(self):
        """테스트 결과 요약 출력"""
        print("\\n" + "="*80)
        print("📊 EDAToolsServerAgent 테스트 결과 요약")
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
        filename = f"eda_tools_server_test_results_{self.test_results['timestamp']}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.test_results, f, indent=2, ensure_ascii=False)
            print(f"💾 테스트 결과 저장: {filename}")
        except Exception as e:
            print(f"⚠️ 결과 저장 실패: {e}")


async def main():
    """메인 테스트 실행 함수"""
    tester = EDAToolsServerTester()
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())