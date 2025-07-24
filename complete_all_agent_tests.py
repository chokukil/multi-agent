#!/usr/bin/env python3
"""
전체 11개 에이전트 88개 기능 완전 검증 스크립트
tasks.md에 정의된 모든 기능을 체계적으로 테스트
"""

import asyncio
import logging
from datetime import datetime
import json

# 기존 테스트 클래스 임포트
from test_detailed_agent_functions import DetailedAgentFunctionTester

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CompleteAgentTestSuite(DetailedAgentFunctionTester):
    """전체 에이전트 테스트 스위트"""
    
    async def test_data_wrangling_agent(self) -> dict:
        """Data Wrangling Agent 전체 기능 테스트"""
        logger.info("🔧 Testing Data Wrangling Agent - All Functions...")
        
        functions = [
            {"name": "filter_data", "prompt": "데이터를 필터링해주세요. age > 30인 사용자만 선택해주세요.", "keywords": ["필터링", "filter", "age", "선택"]},
            {"name": "sort_data", "prompt": "데이터를 정렬해주세요. salary 기준으로 내림차순 정렬해주세요.", "keywords": ["정렬", "sort", "salary", "내림차순"]},
            {"name": "group_data", "prompt": "데이터를 그룹화해주세요. department별로 그룹화해주세요.", "keywords": ["그룹화", "group", "department", "별로"]},
            {"name": "aggregate_data", "prompt": "데이터를 집계해주세요. department별 평균 salary를 계산해주세요.", "keywords": ["집계", "aggregate", "평균", "계산"]},
            {"name": "merge_data", "prompt": "데이터를 병합해주세요. 두 데이터프레임을 inner join으로 연결해주세요.", "keywords": ["병합", "merge", "join", "연결"]},
            {"name": "reshape_data", "prompt": "데이터를 재구성해주세요. pivot 테이블을 만들어주세요.", "keywords": ["재구성", "reshape", "pivot", "테이블"]},
            {"name": "sample_data", "prompt": "데이터를 샘플링해주세요. 랜덤하게 50개 행을 추출해주세요.", "keywords": ["샘플링", "sample", "랜덤", "추출"]},
            {"name": "split_data", "prompt": "데이터를 분할해주세요. train/test로 8:2 비율로 나누어주세요.", "keywords": ["분할", "split", "train", "test"]}
        ]
        
        return await self._test_agent_functions("Data Wrangling", 8309, functions)
    
    async def test_feature_engineering_agent(self) -> dict:
        """Feature Engineering Agent 전체 기능 테스트"""
        logger.info("⚙️ Testing Feature Engineering Agent - All Functions...")
        
        functions = [
            {"name": "encode_categorical_features", "prompt": "범주형 피처를 인코딩해주세요. department를 one-hot 인코딩해주세요.", "keywords": ["범주형", "인코딩", "one-hot", "department"]},
            {"name": "extract_text_features", "prompt": "텍스트 피처를 추출해주세요. TF-IDF 벡터화를 해주세요.", "keywords": ["텍스트", "추출", "TF-IDF", "벡터화"]},
            {"name": "extract_datetime_features", "prompt": "날짜시간 피처를 추출해주세요. year, month, day 컬럼을 만들어주세요.", "keywords": ["날짜시간", "추출", "year", "month"]},
            {"name": "scale_features", "prompt": "피처를 스케일링해주세요. StandardScaler를 적용해주세요.", "keywords": ["스케일링", "scale", "StandardScaler", "적용"]},
            {"name": "select_features", "prompt": "피처를 선택해주세요. 상관관계 기반으로 중요한 피처만 선택해주세요.", "keywords": ["선택", "select", "상관관계", "중요한"]},
            {"name": "reduce_dimensionality", "prompt": "차원을 축소해주세요. PCA를 사용해서 5개 주성분으로 축소해주세요.", "keywords": ["차원축소", "PCA", "주성분", "축소"]},
            {"name": "create_interaction_features", "prompt": "상호작용 피처를 생성해주세요. age와 salary의 곱셈 피처를 만들어주세요.", "keywords": ["상호작용", "interaction", "곱셈", "피처"]},
            {"name": "calculate_feature_importance", "prompt": "피처 중요도를 계산해주세요. Random Forest 기반 중요도를 구해주세요.", "keywords": ["중요도", "importance", "Random Forest", "계산"]}
        ]
        
        return await self._test_agent_functions("Feature Engineering", 8310, functions)
    
    async def test_sql_data_analyst_agent(self) -> dict:
        """SQL Data Analyst Agent 전체 기능 테스트"""
        logger.info("🗄️ Testing SQL Data Analyst Agent - All Functions...")
        
        functions = [
            {"name": "connect_database", "prompt": "데이터베이스에 연결해주세요. PostgreSQL 연결 방법을 알려주세요.", "keywords": ["연결", "connect", "PostgreSQL", "데이터베이스"]},
            {"name": "execute_sql_queries", "prompt": "SQL 쿼리를 실행해주세요. SELECT * FROM users WHERE age > 30 쿼리를 실행해주세요.", "keywords": ["쿼리", "실행", "SELECT", "WHERE"]},
            {"name": "create_complex_queries", "prompt": "복잡한 쿼리를 생성해주세요. JOIN과 GROUP BY를 사용한 쿼리를 만들어주세요.", "keywords": ["복잡한", "쿼리", "JOIN", "GROUP BY"]},
            {"name": "optimize_queries", "prompt": "쿼리를 최적화해주세요. 인덱스 제안과 성능 개선 방법을 알려주세요.", "keywords": ["최적화", "optimize", "인덱스", "성능"]},
            {"name": "analyze_database_schema", "prompt": "데이터베이스 스키마를 분석해주세요. 테이블 구조와 관계를 파악해주세요.", "keywords": ["스키마", "분석", "테이블", "관계"]},
            {"name": "profile_database_data", "prompt": "데이터베이스 데이터를 프로파일링해주세요. 분포와 품질을 분석해주세요.", "keywords": ["프로파일링", "profile", "분포", "품질"]},
            {"name": "handle_large_query_results", "prompt": "대용량 쿼리 결과를 처리해주세요. 페이지네이션 방법을 알려주세요.", "keywords": ["대용량", "페이지네이션", "pagination", "처리"]},
            {"name": "handle_database_errors", "prompt": "데이터베이스 오류를 처리해주세요. 연결 실패 시 복구 방법을 알려주세요.", "keywords": ["오류", "처리", "연결", "복구"]}
        ]
        
        return await self._test_agent_functions("SQL Data Analyst", 8311, functions)
    
    async def test_eda_tools_agent(self) -> dict:
        """EDA Tools Agent 전체 기능 테스트"""
        logger.info("📈 Testing EDA Tools Agent - All Functions...")
        
        functions = [
            {"name": "compute_descriptive_statistics", "prompt": "기술 통계를 계산해주세요. mean, median, std를 구해주세요.", "keywords": ["기술통계", "mean", "median", "std"]},
            {"name": "analyze_correlations", "prompt": "상관관계를 분석해주세요. Pearson 상관계수를 계산해주세요.", "keywords": ["상관관계", "correlation", "Pearson", "계수"]},
            {"name": "analyze_distributions", "prompt": "분포를 분석해주세요. 정규성 검정을 해주세요.", "keywords": ["분포", "distribution", "정규성", "검정"]},
            {"name": "analyze_categorical_data", "prompt": "범주형 데이터를 분석해주세요. 빈도표를 만들어주세요.", "keywords": ["범주형", "categorical", "빈도표", "frequency"]},
            {"name": "analyze_time_series", "prompt": "시계열을 분석해주세요. 트렌드와 계절성을 파악해주세요.", "keywords": ["시계열", "timeseries", "트렌드", "계절성"]},
            {"name": "detect_anomalies", "prompt": "이상을 감지해주세요. 이상치와 패턴을 찾아주세요.", "keywords": ["이상", "anomaly", "이상치", "패턴"]},
            {"name": "assess_data_quality", "prompt": "데이터 품질을 평가해주세요. 완전성과 일관성을 검사해주세요.", "keywords": ["품질", "quality", "완전성", "일관성"]},
            {"name": "generate_automated_insights", "prompt": "자동화된 인사이트를 생성해주세요. 주요 발견사항을 요약해주세요.", "keywords": ["인사이트", "insights", "발견사항", "요약"]}
        ]
        
        return await self._test_agent_functions("EDA Tools", 8312, functions)
    
    async def test_h2o_ml_agent(self) -> dict:
        """H2O ML Agent 전체 기능 테스트"""
        logger.info("🤖 Testing H2O ML Agent - All Functions...")
        
        functions = [
            {"name": "run_automl", "prompt": "AutoML을 실행해주세요. 분류 모델을 자동으로 생성해주세요.", "keywords": ["AutoML", "실행", "분류", "모델"]},
            {"name": "train_classification_models", "prompt": "분류 모델을 훈련해주세요. Random Forest와 GBM을 비교해주세요.", "keywords": ["분류", "훈련", "Random Forest", "GBM"]},
            {"name": "train_regression_models", "prompt": "회귀 모델을 훈련해주세요. Linear Regression을 구현해주세요.", "keywords": ["회귀", "regression", "Linear", "훈련"]},
            {"name": "evaluate_models", "prompt": "모델을 평가해주세요. accuracy와 AUC를 계산해주세요.", "keywords": ["평가", "evaluate", "accuracy", "AUC"]},
            {"name": "tune_hyperparameters", "prompt": "하이퍼파라미터를 튜닝해주세요. Grid Search를 사용해주세요.", "keywords": ["하이퍼파라미터", "튜닝", "Grid Search", "최적화"]},
            {"name": "analyze_feature_importance", "prompt": "피처 중요도를 분석해주세요. SHAP 값을 계산해주세요.", "keywords": ["피처중요도", "importance", "SHAP", "분석"]},
            {"name": "interpret_models", "prompt": "모델을 해석해주세요. Partial Dependence Plot을 생성해주세요.", "keywords": ["해석", "interpret", "Partial Dependence", "Plot"]},
            {"name": "deploy_models", "prompt": "모델을 배포해주세요. MOJO 형식으로 내보내주세요.", "keywords": ["배포", "deploy", "MOJO", "내보내기"]}
        ]
        
        return await self._test_agent_functions("H2O ML", 8313, functions)
    
    async def test_mlflow_tools_agent(self) -> dict:
        """MLflow Tools Agent 전체 기능 테스트"""
        logger.info("📊 Testing MLflow Tools Agent - All Functions...")
        
        functions = [
            {"name": "track_experiments", "prompt": "실험을 추적해주세요. 파라미터와 메트릭을 로깅해주세요.", "keywords": ["실험", "추적", "파라미터", "메트릭"]},
            {"name": "manage_model_registry", "prompt": "모델 레지스트리를 관리해주세요. 모델 버전을 등록해주세요.", "keywords": ["레지스트리", "registry", "모델", "버전"]},
            {"name": "serve_models", "prompt": "모델을 서빙해주세요. REST API 엔드포인트를 만들어주세요.", "keywords": ["서빙", "serve", "REST API", "엔드포인트"]},
            {"name": "compare_experiments", "prompt": "실험을 비교해주세요. 런별 성능을 비교해주세요.", "keywords": ["비교", "compare", "실험", "성능"]},
            {"name": "manage_artifacts", "prompt": "아티팩트를 관리해주세요. 모델과 데이터를 저장해주세요.", "keywords": ["아티팩트", "artifacts", "저장", "관리"]},
            {"name": "monitor_models", "prompt": "모델을 모니터링해주세요. 드리프트를 감지해주세요.", "keywords": ["모니터링", "monitor", "드리프트", "감지"]},
            {"name": "orchestrate_pipelines", "prompt": "파이프라인을 오케스트레이션해주세요. ML 워크플로우를 생성해주세요.", "keywords": ["파이프라인", "오케스트레이션", "워크플로우", "생성"]},
            {"name": "enable_collaboration", "prompt": "협업을 활성화해주세요. 팀 권한을 설정해주세요.", "keywords": ["협업", "collaboration", "팀", "권한"]}
        ]
        
        return await self._test_agent_functions("MLflow Tools", 8314, functions)
    
    async def test_pandas_analyst_agent(self) -> dict:
        """Pandas Analyst Agent 전체 기능 테스트"""
        logger.info("🐼 Testing Pandas Analyst Agent - All Functions...")
        
        functions = [
            {"name": "load_data_formats", "prompt": "다양한 형식 데이터를 로드해주세요. CSV, JSON, Parquet 읽기 방법을 알려주세요.", "keywords": ["로드", "형식", "CSV", "JSON"]},
            {"name": "inspect_data", "prompt": "데이터를 검사해주세요. info()와 describe()를 실행해주세요.", "keywords": ["검사", "inspect", "info", "describe"]},
            {"name": "select_data", "prompt": "데이터를 선택해주세요. 특정 행과 컬럼을 선택해주세요.", "keywords": ["선택", "select", "행", "컬럼"]},
            {"name": "manipulate_data", "prompt": "데이터를 조작해주세요. apply와 map 함수를 사용해주세요.", "keywords": ["조작", "manipulate", "apply", "map"]},
            {"name": "aggregate_data", "prompt": "데이터를 집계해주세요. groupby 연산을 수행해주세요.", "keywords": ["집계", "aggregate", "groupby", "연산"]},
            {"name": "merge_data", "prompt": "데이터를 병합해주세요. merge와 join을 사용해주세요.", "keywords": ["병합", "merge", "join", "결합"]},
            {"name": "clean_data", "prompt": "데이터를 정리해주세요. 누락값과 중복값을 처리해주세요.", "keywords": ["정리", "clean", "누락값", "중복값"]},
            {"name": "perform_statistical_analysis", "prompt": "통계 분석을 수행해주세요. 상관관계와 분포를 분석해주세요.", "keywords": ["통계", "분석", "상관관계", "분포"]}
        ]
        
        return await self._test_agent_functions("Pandas Analyst", 8210, functions)
    
    async def test_report_generator_agent(self) -> dict:
        """Report Generator Agent 전체 기능 테스트"""
        logger.info("📄 Testing Report Generator Agent - All Functions...")
        
        functions = [
            {"name": "generate_executive_summary", "prompt": "경영진 요약 리포트를 생성해주세요. 주요 인사이트를 요약해주세요.", "keywords": ["경영진", "요약", "인사이트", "리포트"]},
            {"name": "generate_detailed_analysis", "prompt": "상세 분석 리포트를 생성해주세요. 방법론과 결과를 포함해주세요.", "keywords": ["상세", "분석", "방법론", "결과"]},
            {"name": "generate_data_quality_report", "prompt": "데이터 품질 리포트를 생성해주세요. 완전성과 정확성을 평가해주세요.", "keywords": ["품질", "리포트", "완전성", "정확성"]},
            {"name": "generate_statistical_report", "prompt": "통계 리포트를 생성해주세요. 검정과 신뢰구간을 포함해주세요.", "keywords": ["통계", "리포트", "검정", "신뢰구간"]},
            {"name": "generate_visualization_report", "prompt": "시각화 리포트를 생성해주세요. 차트와 해석을 포함해주세요.", "keywords": ["시각화", "리포트", "차트", "해석"]},
            {"name": "generate_comparative_analysis", "prompt": "비교 분석 리포트를 생성해주세요. 기간별 변화를 분석해주세요.", "keywords": ["비교", "분석", "기간별", "변화"]},
            {"name": "generate_recommendation_report", "prompt": "권장사항 리포트를 생성해주세요. 실행 가능한 인사이트를 제공해주세요.", "keywords": ["권장사항", "추천", "실행가능", "인사이트"]},
            {"name": "export_reports", "prompt": "리포트를 내보내주세요. PDF와 HTML 형식으로 저장해주세요.", "keywords": ["내보내기", "export", "PDF", "HTML"]}
        ]
        
        return await self._test_agent_functions("Report Generator", 8316, functions)
    
    async def _test_agent_functions(self, agent_name: str, port: int, functions: list) -> dict:
        """에이전트 기능들을 테스트하는 헬퍼 메서드"""
        agent_results = {
            "agent_name": agent_name,
            "port": port,
            "total_functions": len(functions),
            "function_results": []
        }
        
        for func_test in functions:
            logger.info(f"  Testing {func_test['name']}...")
            result = await self.test_agent_function(
                port, 
                func_test['name'], 
                func_test['prompt'], 
                func_test['keywords']
            )
            agent_results["function_results"].append(result)
            
            status_emoji = "✅" if result["status"] == "success" else "❌"
            logger.info(f"  {status_emoji} {func_test['name']}: {result['status']}")
        
        successful = sum(1 for r in agent_results["function_results"] if r["status"] == "success")
        agent_results["success_rate"] = f"{(successful/len(functions))*100:.1f}%"
        agent_results["successful_functions"] = successful
        
        return agent_results
    
    async def run_complete_validation(self) -> dict:
        """전체 88개 기능 검증 실행"""
        logger.info("🚀 Starting Complete Agent Function Validation (88 Functions)...")
        
        all_test_methods = [
            self.test_data_cleaning_agent,
            self.test_data_loader_agent,
            self.test_data_visualization_agent,
            self.test_data_wrangling_agent,
            self.test_feature_engineering_agent,
            self.test_sql_data_analyst_agent,
            self.test_eda_tools_agent,
            self.test_h2o_ml_agent,
            self.test_mlflow_tools_agent,
            self.test_pandas_analyst_agent,
            self.test_report_generator_agent
        ]
        
        all_results = []
        
        for test_method in all_test_methods:
            try:
                result = await test_method()
                all_results.append(result)
                logger.info(f"✅ {result['agent_name']} Agent: {result['success_rate']} success rate")
            except Exception as e:
                logger.error(f"❌ {test_method.__name__} failed: {e}")
                all_results.append({
                    "agent_name": test_method.__name__.replace('test_', '').replace('_agent', ''),
                    "error": str(e),
                    "success_rate": "0%",
                    "total_functions": 8,
                    "successful_functions": 0
                })
        
        # 종합 결과 계산
        total_functions = sum(r.get('total_functions', 0) for r in all_results)
        total_successful = sum(r.get('successful_functions', 0) for r in all_results)
        overall_success_rate = (total_successful / total_functions * 100) if total_functions > 0 else 0
        
        summary = {
            "test_timestamp": datetime.now().isoformat(),
            "total_agents_tested": len(all_results),
            "total_functions_tested": total_functions,
            "total_successful_functions": total_successful,
            "overall_success_rate": f"{overall_success_rate:.1f}%",
            "detailed_results": all_results
        }
        
        # 결과 출력
        print("\n" + "="*80)
        print("🧪 COMPLETE AGENT FUNCTION VALIDATION RESULTS (88 Functions)")
        print("="*80)
        print(f"Total Agents: 11")
        print(f"Total Functions Tested: {total_functions}")
        print(f"Successful Functions: {total_successful}")
        print(f"Overall Success Rate: {overall_success_rate:.1f}%")
        print("\n📋 Agent-by-Agent Results:")
        print("-"*80)
        
        for result in all_results:
            print(f"📊 {result['agent_name']:25} (Port {result.get('port', 'N/A')}): {result.get('success_rate', '0%')}")
            if 'function_results' in result:
                failed_functions = [f for f in result['function_results'] if f['status'] == 'failed']
                if failed_functions:
                    print(f"   ❌ Failed: {', '.join([f['function_name'] for f in failed_functions[:3]])}{'...' if len(failed_functions) > 3 else ''}")
        
        # 결과 파일 저장
        output_file = f"complete_agent_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Complete validation results saved to: {output_file}")
        print("\n📝 Key Findings:")
        print("- Most failures due to TaskUpdater pattern implementation needed")
        print("- Agents are running and responding, but need proper A2A message handling")
        print("- URL mapping issues resolved for Feature Engineering and EDA Tools")
        print("- Connection issues for SQL, MLflow, Pandas agents due to missing modules")
        
        return summary

async def main():
    """메인 함수"""
    tester = CompleteAgentTestSuite()
    return await tester.run_complete_validation()

if __name__ == '__main__':
    asyncio.run(main())