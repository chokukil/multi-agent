#!/usr/bin/env python3
"""
DataWranglingServerAgent 테스트 스크립트

새로운 data_wrangling_server_new.py가 올바르게 작동하는지 확인합니다.
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

from a2a_ds_servers.data_wrangling_server_new import DataWranglingServerAgent

class DataWranglingServerTester:
    """DataWranglingServerAgent 테스트 클래스"""
    
    def __init__(self):
        self.test_results = {
            "timestamp": time.strftime("%Y%m%d_%H%M%S"),
            "tests": []
        }
    
    async def test_initialization(self):
        """서버 에이전트 초기화 테스트"""
        print("🔧 DataWranglingServerAgent 초기화 테스트")
        
        try:
            agent = DataWranglingServerAgent()
            
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
        
        # 랭글링에 적합한 CSV 데이터 테스트
        csv_data = """id,name,category,sales,region
1,Product A,Electronics,1000,North
2,Product B,Clothing,800,South
3,Product C,Electronics,1200,North
4,Product D,Books,600,East
5,Product E,Clothing,900,South"""
        
        df = agent.data_processor.parse_data_from_message(csv_data)
        
        if df is not None:
            print(f"   ✅ CSV 파싱 성공: {df.shape}")
            print(f"   📊 숫자형 컬럼: {len(df.select_dtypes(include=['number']).columns)}개")
            print(f"   📝 범주형 컬럼: {len(df.select_dtypes(include=['object']).columns)}개")
            print(f"   🏷️ 카테고리 종류: {df['category'].nunique() if 'category' in df.columns else 0}개")
            test_result = {
                "test_name": "data_parsing_csv",
                "status": "PASS", 
                "details": {
                    "shape": df.shape, 
                    "columns": list(df.columns),
                    "numeric_cols": len(df.select_dtypes(include=['number']).columns),
                    "categorical_cols": len(df.select_dtypes(include=['object']).columns)
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
    
    async def test_data_wrangling_process(self, agent):
        """데이터 랭글링 프로세스 테스트"""
        print("\\n🔧 데이터 랭글링 프로세스 테스트")
        
        # 테스트 데이터와 요청
        test_request = """다음 데이터를 카테고리별로 그룹화해서 매출 평균을 계산해주세요:

id,name,category,sales,region
1,Product A,Electronics,1000,North
2,Product B,Clothing,800,South
3,Product C,Electronics,1200,North
4,Product D,Books,600,East
5,Product E,Clothing,900,South
6,Product F,Electronics,1100,West"""
        
        try:
            result = await agent.process_data_wrangling(test_request)
            
            if result and len(result) > 100:
                print("   ✅ 데이터 랭글링 성공")
                print(f"   📄 결과 길이: {len(result)} 문자")
                
                # 랭글링 특화 키워드 검증
                wrangling_keywords = ["DataWranglingAgent", "랭글링", "그룹", "집계", "Complete"]
                found_keywords = [kw for kw in wrangling_keywords if kw in result]
                print(f"   🔍 랭글링 키워드 발견: {len(found_keywords)}/{len(wrangling_keywords)}")
                
                # 집계 관련 키워드 확인
                agg_keywords = ["group", "평균", "average", "mean", "category"]
                found_agg_keywords = [kw for kw in agg_keywords if kw.lower() in result.lower()]
                print(f"   📊 집계 키워드 발견: {len(found_agg_keywords)}/{len(agg_keywords)}")
                
                test_result = {
                    "test_name": "data_wrangling_process",
                    "status": "PASS",
                    "details": {
                        "result_length": len(result),
                        "wrangling_keywords_found": found_keywords,
                        "agg_keywords_found": found_agg_keywords
                    }
                }
            else:
                print("   ❌ 데이터 랭글링 실패 - 결과 부족")
                test_result = {
                    "test_name": "data_wrangling_process",
                    "status": "FAIL",
                    "reason": "insufficient_result"
                }
            
            self.test_results["tests"].append(test_result)
            return result
            
        except Exception as e:
            print(f"   ❌ 데이터 랭글링 오류: {e}")
            test_result = {
                "test_name": "data_wrangling_process",
                "status": "ERROR",
                "error": str(e)
            }
            self.test_results["tests"].append(test_result)
            return None
    
    async def test_merge_functionality(self, agent):
        """데이터 병합 기능 테스트"""
        print("\\n🔗 데이터 병합 기능 테스트")
        
        # 병합 시나리오 테스트
        merge_request = """다음 두 데이터셋을 ID를 기준으로 병합해주세요:

customers:
id,name,city
1,Alice,Seoul
2,Bob,Busan
3,Charlie,Daegu

orders:
customer_id,product,amount
1,Laptop,1000
2,Phone,800
1,Tablet,600"""
        
        try:
            result = await agent.process_data_wrangling(merge_request)
            
            if result and ("병합" in result or "merge" in result.lower()):
                print("   ✅ 데이터 병합 성공")
                
                # 병합 관련 키워드 확인
                merge_keywords = ["join", "merge", "병합", "customer", "id"]
                found_merge_keywords = [kw for kw in merge_keywords if kw.lower() in result.lower()]
                print(f"   🔗 병합 키워드 발견: {len(found_merge_keywords)}/{len(merge_keywords)}")
                
                test_result = {
                    "test_name": "merge_functionality",
                    "status": "PASS",
                    "details": {
                        "merge_keywords": found_merge_keywords
                    }
                }
            else:
                print("   ❌ 데이터 병합 실패")
                test_result = {
                    "test_name": "merge_functionality", 
                    "status": "FAIL"
                }
            
            self.test_results["tests"].append(test_result)
            
        except Exception as e:
            print(f"   ❌ 데이터 병합 오류: {e}")
            test_result = {
                "test_name": "merge_functionality",
                "status": "ERROR",
                "error": str(e)
            }
            self.test_results["tests"].append(test_result)
    
    async def test_guidance_generation(self, agent):
        """가이드 생성 테스트"""
        print("\\n📚 가이드 생성 테스트")
        
        try:
            guidance = await agent.process_data_wrangling("데이터 랭글링 방법을 알려주세요")
            
            if guidance and "가이드" in guidance:
                print("   ✅ 가이드 생성 성공")
                
                # 가이드 특화 키워드 확인
                guide_keywords = ["랭글링", "병합", "집계", "변환", "인코딩"]
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
        print("🚀 DataWranglingServerAgent 종합 테스트 시작\\n")
        
        # 초기화 테스트
        agent = await self.test_initialization()
        if not agent:
            print("❌ 초기화 실패로 테스트 중단")
            return
        
        # 데이터 파싱 테스트
        df = await self.test_data_parsing(agent)
        
        # 데이터 랭글링 프로세스 테스트
        result = await self.test_data_wrangling_process(agent)
        
        # 데이터 병합 테스트
        await self.test_merge_functionality(agent)
        
        # 가이드 생성 테스트
        await self.test_guidance_generation(agent)
        
        # 결과 요약
        self.print_test_summary()
        self.save_test_results()
    
    def print_test_summary(self):
        """테스트 결과 요약 출력"""
        print("\\n" + "="*80)
        print("🔧 DataWranglingServerAgent 테스트 결과 요약")
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
        filename = f"data_wrangling_server_test_results_{self.test_results['timestamp']}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.test_results, f, indent=2, ensure_ascii=False)
            print(f"💾 테스트 결과 저장: {filename}")
        except Exception as e:
            print(f"⚠️ 결과 저장 실패: {e}")


async def main():
    """메인 테스트 실행 함수"""
    tester = DataWranglingServerTester()
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())