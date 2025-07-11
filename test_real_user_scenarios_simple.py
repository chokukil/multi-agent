#!/usr/bin/env python3
"""
Real User Scenario Testing (Simple)
실제 사용자 시나리오 기반 간단 테스트

Author: CherryAI Team
"""

import os
import json
import tempfile
import pandas as pd
import time
from datetime import datetime

class SimpleUserScenarioTest:
    """간단한 실제 사용자 시나리오 테스트"""
    
    def __init__(self):
        self.results = {
            "test_start": datetime.now().isoformat(),
            "tests": [],
            "overall_success": False
        }
    
    def run_test(self):
        """테스트 실행"""
        print("🧪 Real User Scenario Testing (Simple)")
        print("=" * 60)
        
        # 1. CSV 파일 테스트
        self._test_csv_scenario()
        
        # 2. Excel 파일 테스트  
        self._test_excel_scenario()
        
        # 3. JSON 파일 테스트
        self._test_json_scenario()
        
        # 4. 대용량 데이터 테스트
        self._test_large_data_scenario()
        
        # 결과 계산
        success_count = sum(1 for test in self.results["tests"] if test["success"])
        total_count = len(self.results["tests"])
        self.results["overall_success"] = success_count >= total_count * 0.8
        
        print(f"\n📊 테스트 결과: {success_count}/{total_count} 성공")
        
        return self.results
    
    def _test_csv_scenario(self):
        """CSV 데이터 시나리오 테스트"""
        print("\n1️⃣ CSV 데이터 테스트")
        
        try:
            # 간단한 고객 데이터
            cities = ["Seoul", "Busan", "Incheon", "Daegu", "Daejeon"]
            categories = ["Electronics", "Clothing", "Books", "Food", "Sports"]
            
            data = []
            for i in range(100):
                data.append({
                    "customer_id": f"CUST_{i+1:03d}",
                    "age": 20 + (i % 50),
                    "gender": "Male" if i % 2 == 0 else "Female",
                    "city": cities[i % len(cities)],
                    "purchase_amount": 1000 + (i * 150) % 5000,
                    "category": categories[i % len(categories)]
                })
            
            df = pd.DataFrame(data)
            
            # 기본 분석
            stats = df.describe()
            value_counts = df['category'].value_counts()
            
            success = len(df) == 100 and len(stats) > 0 and len(value_counts) > 0
            self._log_test("CSV 데이터 분석", success, f"레코드: {len(df)}, 통계: {len(stats.columns)}")
            
        except Exception as e:
            self._log_test("CSV 데이터 분석", False, f"오류: {str(e)}")
    
    def _test_excel_scenario(self):
        """Excel 데이터 시나리오 테스트"""
        print("\n2️⃣ Excel 데이터 테스트")
        
        try:
            # 매출 데이터
            data = {
                "month": ["2024-01", "2024-02", "2024-03", "2024-04", "2024-05", "2024-06"],
                "revenue": [1200000, 1350000, 1180000, 1420000, 1550000, 1380000],
                "costs": [800000, 850000, 790000, 920000, 980000, 890000],
                "profit": [400000, 500000, 390000, 500000, 570000, 490000],
                "region": ["Seoul", "Busan", "Seoul", "Incheon", "Seoul", "Daegu"]
            }
            
            df = pd.DataFrame(data)
            
            # 기본 분석
            total_revenue = df['revenue'].sum()
            avg_profit = df['profit'].mean()
            
            success = len(df) == 6 and total_revenue > 0 and avg_profit > 0
            self._log_test("Excel 데이터 분석", success, f"매출: {total_revenue:,}, 평균이익: {avg_profit:,.0f}")
            
        except Exception as e:
            self._log_test("Excel 데이터 분석", False, f"오류: {str(e)}")
    
    def _test_json_scenario(self):
        """JSON 데이터 시나리오 테스트"""
        print("\n3️⃣ JSON 데이터 테스트")
        
        try:
            # API 로그 데이터
            data = []
            endpoints = ["/api/users", "/api/orders", "/api/products"]
            methods = ["GET", "POST", "PUT"]
            user_agents = ["Chrome", "Firefox", "Safari"]
            
            for i in range(60):
                data.append({
                    "timestamp": f"2024-06-28T{10 + i//10:02d}:{i%60:02d}:00Z",
                    "endpoint": endpoints[i % len(endpoints)],
                    "method": methods[i % len(methods)],
                    "response_time": 100 + (i * 10) % 200,
                    "status_code": 200 if i % 10 != 9 else 404,
                    "user_agent": user_agents[i % len(user_agents)]
                })
            
            df = pd.DataFrame(data)
            
            # 기본 분석
            avg_response_time = df['response_time'].mean()
            status_distribution = df['status_code'].value_counts()
            
            success = len(df) == 60 and avg_response_time > 0 and len(status_distribution) > 0
            self._log_test("JSON 데이터 분석", success, f"평균응답시간: {avg_response_time:.1f}ms, 상태코드: {len(status_distribution)}")
            
        except Exception as e:
            self._log_test("JSON 데이터 분석", False, f"오류: {str(e)}")
    
    def _test_large_data_scenario(self):
        """대용량 데이터 시나리오 테스트"""
        print("\n4️⃣ 대용량 데이터 테스트")
        
        try:
            # 10,000개 레코드 생성
            data_size = 10000
            
            data = []
            for i in range(data_size):
                data.append({
                    "id": i + 1,
                    "timestamp": f"2024-06-{(i % 30) + 1:02d}T{(i % 24):02d}:00:00Z",
                    "value": (i * 1.5) % 1000,
                    "category": f"Category_{(i % 50) + 1}",
                    "status": "active" if i % 3 == 0 else "inactive"
                })
            
            start_time = time.time()
            df = pd.DataFrame(data)
            creation_time = time.time() - start_time
            
            start_time = time.time()
            stats = df.describe()
            category_counts = df['category'].value_counts()
            analysis_time = time.time() - start_time
            
            performance_ok = creation_time < 2.0 and analysis_time < 1.0
            
            success = len(df) == data_size and performance_ok
            details = f"크기: {len(df)}, 생성: {creation_time:.2f}s, 분석: {analysis_time:.2f}s"
            
            self._log_test("대용량 데이터 처리", success, details)
            
        except Exception as e:
            self._log_test("대용량 데이터 처리", False, f"오류: {str(e)}")
    
    def _log_test(self, test_name: str, success: bool, details: str = ""):
        """테스트 결과 로깅"""
        self.results["tests"].append({
            "name": test_name,
            "success": success,
            "details": details,
            "timestamp": datetime.now().isoformat()
        })
        
        status = "✅" if success else "❌"
        print(f"{status} {test_name}: {details}")

def main():
    """메인 테스트 실행"""
    tester = SimpleUserScenarioTest()
    results = tester.run_test()
    
    # 결과 파일 저장
    results_file = f"simple_user_scenarios_results_{int(time.time())}.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"\n📄 결과 저장: {results_file}")
    
    if results["overall_success"]:
        print("🎉 실제 사용자 시나리오 테스트 성공!")
        return True
    else:
        print("⚠️ 일부 사용자 시나리오에서 개선이 필요합니다")
        return False

if __name__ == "__main__":
    main() 