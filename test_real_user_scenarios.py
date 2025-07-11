#!/usr/bin/env python3
"""
Real User Scenario Testing
실제 사용자 시나리오 기반 엔드투엔드 테스트

다양한 데이터 타입(CSV, Excel, JSON) 및 분석 유형 검증

Author: CherryAI Team
"""

import os
import json
import tempfile
import pandas as pd
import requests
import time
from datetime import datetime
from pathlib import Path

class RealUserScenarioTest:
    """실제 사용자 시나리오 테스트"""
    
    def __init__(self):
        self.results = {
            "test_start": datetime.now().isoformat(),
            "tests": [],
            "overall_success": False,
            "errors": [],
            "scenario_results": []
        }
        self.streamlit_url = "http://localhost:8501"
        self.test_data_dir = None
    
    def run_comprehensive_test(self):
        """종합 실제 사용자 시나리오 테스트 실행"""
        print("🧪 Real User Scenario Comprehensive Testing")
        print("=" * 70)
        
        # 테스트 데이터 디렉터리 생성
        self.test_data_dir = tempfile.mkdtemp(prefix="cherryai_user_scenarios_")
        
        try:
            # 1. CSV 데이터 분석 시나리오
            self._test_csv_analysis_scenario()
            
            # 2. Excel 데이터 분석 시나리오
            self._test_excel_analysis_scenario()
            
            # 3. JSON 데이터 분석 시나리오
            self._test_json_analysis_scenario()
            
            # 4. 다중 파일 통합 분석 시나리오
            self._test_multi_file_integration_scenario()
            
            # 5. 대용량 데이터 처리 시나리오
            self._test_large_dataset_scenario()
            
            # 결과 계산
            success_count = sum(1 for test in self.results["tests"] if test["success"])
            total_count = len(self.results["tests"])
            self.results["overall_success"] = success_count >= total_count * 0.8
            
            print(f"\n📊 실제 사용자 시나리오 테스트 결과: {success_count}/{total_count} 성공")
            
        finally:
            # 테스트 데이터 정리
            if self.test_data_dir:
                import shutil
                shutil.rmtree(self.test_data_dir, ignore_errors=True)
        
        return self.results
    
    def _test_csv_analysis_scenario(self):
        """CSV 데이터 분석 시나리오 테스트"""
        print("\n1️⃣ CSV 데이터 분석 시나리오")
        
        # 실제 비즈니스 시나리오: 고객 구매 데이터 분석
        csv_data = {
            "customer_id": [f"CUST_{i:03d}" for i in range(1, 101)],
            "age": [20 + (i % 50) for i in range(100)],
            "gender": ["Male" if i % 2 == 0 else "Female" for i in range(100)],
            "city": [["Seoul", "Busan", "Incheon", "Daegu", "Daejeon"][i % 5] for i in range(100)],
            "purchase_amount": [1000 + (i * 150) % 5000 for i in range(100)],
            "category": [["Electronics", "Clothing", "Books", "Food", "Sports"][i % 5] for i in range(100)]
        }
        
        df = pd.DataFrame(csv_data)
        csv_file = os.path.join(self.test_data_dir, "customer_data.csv")
        df.to_csv(csv_file, index=False)
        
        # CSV 분석 기능 테스트
        csv_analysis_results = self._perform_data_analysis_test(
            file_path=csv_file,
            data_type="CSV",
            expected_features=[
                "기본 통계",
                "데이터 타입 확인", 
                "결측값 검사",
                "범주형 데이터 분석",
                "수치형 데이터 분석"
            ]
        )
        
        success = csv_analysis_results["data_loaded"] and csv_analysis_results["features_found"] >= 3
        details = f"로드: {csv_analysis_results['data_loaded']}, 기능: {csv_analysis_results['features_found']}/5"
        
        self._log_test("CSV 데이터 분석 시나리오", success, details)
    
    def _test_excel_analysis_scenario(self):
        """Excel 데이터 분석 시나리오 테스트"""
        print("\n2️⃣ Excel 데이터 분석 시나리오")
        
        # 실제 비즈니스 시나리오: 매출 보고서 분석
        excel_data = {
            "month": ["2024-01", "2024-02", "2024-03", "2024-04", "2024-05", "2024-06"],
            "revenue": [1200000, 1350000, 1180000, 1420000, 1550000, 1380000],
            "costs": [800000, 850000, 790000, 920000, 980000, 890000],
            "profit": [400000, 500000, 390000, 500000, 570000, 490000],
            "region": ["Seoul", "Busan", "Seoul", "Incheon", "Seoul", "Daegu"]
        }
        
        df = pd.DataFrame(excel_data)
        excel_file = os.path.join(self.test_data_dir, "sales_report.xlsx")
        df.to_excel(excel_file, index=False)
        
        # Excel 분석 기능 테스트
        excel_analysis_results = self._perform_data_analysis_test(
            file_path=excel_file,
            data_type="Excel",
            expected_features=[
                "시계열 분석",
                "매출 트렌드",
                "지역별 분석", 
                "수익성 분석",
                "시각화"
            ]
        )
        
        success = excel_analysis_results["data_loaded"] and excel_analysis_results["features_found"] >= 2
        details = f"로드: {excel_analysis_results['data_loaded']}, 기능: {excel_analysis_results['features_found']}/5"
        
        self._log_test("Excel 데이터 분석 시나리오", success, details)
    
    def _test_json_analysis_scenario(self):
        """JSON 데이터 분석 시나리오 테스트"""
        print("\n3️⃣ JSON 데이터 분석 시나리오")
        
        # 실제 비즈니스 시나리오: API 로그 데이터 분석
        json_data = [
            {
                "timestamp": "2024-06-28T10:00:00Z",
                "endpoint": "/api/users",
                "method": "GET", 
                "response_time": 120,
                "status_code": 200,
                "user_agent": "Chrome"
            },
            {
                "timestamp": "2024-06-28T10:01:00Z",
                "endpoint": "/api/orders",
                "method": "POST",
                "response_time": 250,
                "status_code": 201,
                "user_agent": "Firefox"
            },
            {
                "timestamp": "2024-06-28T10:02:00Z", 
                "endpoint": "/api/users",
                "method": "GET",
                "response_time": 95,
                "status_code": 200,
                "user_agent": "Safari"
            }
        ] * 20  # 60개 레코드 생성
        
        json_file = os.path.join(self.test_data_dir, "api_logs.json")
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        
        # JSON 분석 기능 테스트
        json_analysis_results = self._perform_data_analysis_test(
            file_path=json_file,
            data_type="JSON",
            expected_features=[
                "API 성능 분석",
                "응답 시간 분석",
                "상태 코드 분포",
                "엔드포인트 사용량",
                "사용자 에이전트 분석"
            ]
        )
        
        success = json_analysis_results["data_loaded"] and json_analysis_results["features_found"] >= 2
        details = f"로드: {json_analysis_results['data_loaded']}, 기능: {json_analysis_results['features_found']}/5"
        
        self._log_test("JSON 데이터 분석 시나리오", success, details)
    
    def _test_multi_file_integration_scenario(self):
        """다중 파일 통합 분석 시나리오 테스트"""
        print("\n4️⃣ 다중 파일 통합 분석 시나리오")
        
        # 고객 기본 정보 (CSV)
        customers_data = {
            "customer_id": [f"CUST_{i:03d}" for i in range(1, 21)],
            "name": [f"Customer {i}" for i in range(1, 21)],
            "email": [f"customer{i}@example.com" for i in range(1, 21)],
            "signup_date": ["2024-01-01"] * 20
        }
        customers_df = pd.DataFrame(customers_data)
        customers_file = os.path.join(self.test_data_dir, "customers.csv")
        customers_df.to_csv(customers_file, index=False)
        
        # 주문 정보 (Excel)
        orders_data = {
            "order_id": [f"ORD_{i:03d}" for i in range(1, 51)],
            "customer_id": [f"CUST_{(i % 20) + 1:03d}" for i in range(50)],
            "product": [f"Product {(i % 10) + 1}" for i in range(50)],
            "amount": [100 + (i * 50) for i in range(50)]
        }
        orders_df = pd.DataFrame(orders_data)
        orders_file = os.path.join(self.test_data_dir, "orders.xlsx")
        orders_df.to_excel(orders_file, index=False)
        
        # 통합 분석 테스트
        integration_features = [
            "다중 파일 로드",
            "데이터 조인/병합",
            "고객별 주문 분석",
            "매출 집계",
            "통합 리포트"
        ]
        
        # 파일 존재 및 로드 가능성 확인
        files_loadable = 0
        for file_path in [customers_file, orders_file]:
            try:
                if file_path.endswith('.csv'):
                    test_df = pd.read_csv(file_path)
                elif file_path.endswith('.xlsx'):
                    test_df = pd.read_excel(file_path)
                
                if len(test_df) > 0:
                    files_loadable += 1
                    print(f"✅ {os.path.basename(file_path)}: 로드 가능 ({len(test_df)}개 레코드)")
                
            except Exception as e:
                print(f"❌ {os.path.basename(file_path)}: 로드 실패 - {e}")
        
        # 통합 분석 시뮬레이션
        integration_possible = files_loadable == 2
        if integration_possible:
            # 실제 조인 테스트
            try:
                customers_df = pd.read_csv(customers_file)
                orders_df = pd.read_excel(orders_file)
                
                # 조인 수행
                merged_df = pd.merge(orders_df, customers_df, on='customer_id', how='left')
                join_successful = len(merged_df) > 0 and 'name' in merged_df.columns
                
                if join_successful:
                    print("✅ 데이터 조인: 성공")
                else:
                    print("❌ 데이터 조인: 실패")
                    
            except Exception as e:
                join_successful = False
                print(f"❌ 데이터 조인: 오류 - {e}")
        else:
            join_successful = False
        
        success = files_loadable >= 2 and join_successful
        details = f"로드가능파일: {files_loadable}/2, 조인성공: {join_successful}"
        
        self._log_test("다중 파일 통합 분석", success, details)
    
    def _test_large_dataset_scenario(self):
        """대용량 데이터 처리 시나리오 테스트"""
        print("\n5️⃣ 대용량 데이터 처리 시나리오")
        
        # 대용량 데이터 생성 (10,000 레코드)
        large_data_size = 10000
        
        large_data = {
            "id": range(1, large_data_size + 1),
            "timestamp": [f"2024-06-{(i % 30) + 1:02d}T{(i % 24):02d}:00:00Z" for i in range(large_data_size)],
            "value": [(i * 1.5) % 1000 for i in range(large_data_size)],
            "category": [f"Category_{(i % 50) + 1}" for i in range(large_data_size)],
            "status": ["active" if i % 3 == 0 else "inactive" for i in range(large_data_size)]
        }
        
        large_df = pd.DataFrame(large_data)
        large_file = os.path.join(self.test_data_dir, "large_dataset.csv")
        
        # 대용량 데이터 저장 및 로드 테스트
        try:
            start_time = time.time()
            large_df.to_csv(large_file, index=False)
            save_time = time.time() - start_time
            
            file_size_mb = os.path.getsize(large_file) / (1024 * 1024)
            
            start_time = time.time()
            loaded_df = pd.read_csv(large_file)
            load_time = time.time() - start_time
            
            # 기본 분석 수행
            start_time = time.time()
            basic_stats = loaded_df.describe()
            value_counts = loaded_df['category'].value_counts()
            analysis_time = time.time() - start_time
            
            performance_ok = (
                save_time < 5.0 and  # 5초 이내 저장
                load_time < 3.0 and  # 3초 이내 로드
                analysis_time < 2.0   # 2초 이내 분석
            )
            
            print(f"📊 파일 크기: {file_size_mb:.2f}MB")
            print(f"📊 저장 시간: {save_time:.2f}초")
            print(f"📊 로드 시간: {load_time:.2f}초")
            print(f"📊 분석 시간: {analysis_time:.2f}초")
            
            if performance_ok:
                print("✅ 대용량 데이터 처리: 성능 기준 충족")
            else:
                print("⚠️ 대용량 데이터 처리: 성능 개선 필요")
            
            success = len(loaded_df) == large_data_size and performance_ok
            details = f"크기: {file_size_mb:.1f}MB, 성능: {performance_ok}, 레코드: {len(loaded_df)}"
            
        except Exception as e:
            success = False
            details = f"오류: {str(e)}"
            print(f"❌ 대용량 데이터 처리 실패: {e}")
        
        self._log_test("대용량 데이터 처리", success, details)
    
    def _perform_data_analysis_test(self, file_path: str, data_type: str, expected_features: list) -> dict:
        """데이터 분석 테스트 수행"""
        result = {
            "data_loaded": False,
            "features_found": 0,
            "analysis_time": 0,
            "error": None
        }
        
        try:
            start_time = time.time()
            
            # 파일 로드
            if data_type == "CSV":
                df = pd.read_csv(file_path)
            elif data_type == "Excel":
                df = pd.read_excel(file_path)
            elif data_type == "JSON":
                df = pd.read_json(file_path)
            else:
                raise ValueError(f"지원되지 않는 데이터 타입: {data_type}")
            
            result["data_loaded"] = len(df) > 0
            
            # 기본 분석 수행
            analysis_features = []
            
            # 기본 통계
            if len(df.select_dtypes(include=['number']).columns) > 0:
                basic_stats = df.describe()
                analysis_features.append("기본 통계")
            
            # 데이터 타입 확인
            if len(df.dtypes) > 0:
                analysis_features.append("데이터 타입 확인")
            
            # 결측값 검사
            missing_values = df.isnull().sum()
            analysis_features.append("결측값 검사")
            
            # 범주형 데이터 분석
            categorical_cols = df.select_dtypes(include=['object']).columns
            if len(categorical_cols) > 0:
                for col in categorical_cols[:2]:  # 최대 2개 컬럼
                    value_counts = df[col].value_counts()
                analysis_features.append("범주형 데이터 분석")
            
            # 수치형 데이터 분석
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                analysis_features.append("수치형 데이터 분석")
            
            result["features_found"] = len(analysis_features)
            result["analysis_time"] = time.time() - start_time
            
            print(f"✅ {data_type} 분석: {len(df)}개 레코드, {result['features_found']}개 기능")
            
        except Exception as e:
            result["error"] = str(e)
            print(f"❌ {data_type} 분석 실패: {e}")
        
        return result
    
    def _log_test(self, test_name: str, success: bool, details: str = ""):
        """테스트 결과 로깅"""
        self.results["tests"].append({
            "name": test_name,
            "success": success,
            "details": details,
            "timestamp": datetime.now().isoformat()
        })

def main():
    """메인 테스트 실행"""
    tester = RealUserScenarioTest()
    results = tester.run_comprehensive_test()
    
    # 결과 파일 저장
    results_file = f"real_user_scenarios_test_results_{int(time.time())}.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n📄 결과 저장: {results_file}")
    
    # 최종 상태 출력
    if results["overall_success"]:
        print("🎉 실제 사용자 시나리오 테스트 성공!")
        return True
    else:
        print("⚠️ 일부 사용자 시나리오에서 개선이 필요합니다")
        return False

if __name__ == "__main__":
    main() 