"""
🔍 Enhanced Langfuse Tracking Test Script
계층적 span 구조와 AI-Data-Science-Team 내부 처리 과정 추적 테스트

이 스크립트는 다음을 테스트합니다:
- 계층적 span 구조
- AI-Data-Science-Team 내부 처리 단계별 추적
- LLM 프롬프트/응답 아티팩트 저장
- 코드 생성 및 실행 과정 추적
- 데이터 변환 과정 가시화
"""

import asyncio
import time
import pandas as pd
import json
from datetime import datetime

try:
    from core.langfuse_session_tracer import SessionBasedTracer, get_session_tracer
    from core.langfuse_ai_ds_team_wrapper import LangfuseAIDataScienceTeamWrapper
    LANGFUSE_AVAILABLE = True
except ImportError as e:
    print(f"❌ Langfuse 모듈을 가져올 수 없습니다: {e}")
    LANGFUSE_AVAILABLE = False


class EnhancedTrackingDemo:
    """Enhanced Langfuse 추적 시스템 데모"""
    
    def __init__(self):
        self.session_tracer = None
        self.ai_ds_wrapper = None
        
    def initialize_langfuse(self):
        """Langfuse 세션 초기화"""
        if not LANGFUSE_AVAILABLE:
            print("❌ Langfuse가 사용 불가능합니다.")
            return False
            
        try:
            # 세션 기반 tracer 초기화
            self.session_tracer = get_session_tracer()
            if not self.session_tracer:
                print("❌ Session tracer를 생성할 수 없습니다.")
                return False
                
            # 세션 시작
            session_id = f"enhanced_tracking_test_{int(time.time())}"
            metadata = {
                "test_type": "enhanced_tracking",
                "timestamp": datetime.now().isoformat(),
                "demo_version": "v2.0_enhanced",
                "tracking_features": [
                    "nested_spans",
                    "ai_ds_team_workflow",
                    "llm_step_tracking", 
                    "code_generation_tracking",
                    "data_transformation_tracking",
                    "artifact_storage"
                ]
            }
            
            self.session_tracer.start_user_session(
                "Enhanced tracking test query", 
                "enhanced_demo_user", 
                metadata
            )
            print(f"✅ Langfuse 세션 시작: {session_id}")
            
            # AI-Data-Science-Team wrapper 생성
            self.ai_ds_wrapper = LangfuseAIDataScienceTeamWrapper(
                self.session_tracer, 
                "Enhanced Demo Agent"
            )
            
            return True
            
        except Exception as e:
            print(f"❌ Langfuse 초기화 실패: {e}")
            return False
    
    def simulate_ai_ds_team_workflow(self):
        """AI-Data-Science-Team 워크플로우 시뮬레이션"""
        print("\n🔍 AI-Data-Science-Team 워크플로우 시뮬레이션 시작")
        
        # 1. 메인 agent span 생성
        operation_data = {
            "operation": "data_cleaning",
            "user_request": "결측값과 이상값을 처리해주세요",
            "data_source": "sample_dataset.csv"
        }
        
        main_span = self.ai_ds_wrapper.create_agent_span("Enhanced Data Cleaning", operation_data)
        
        # 2. 워크플로우 시작 추적
        self.ai_ds_wrapper.trace_ai_ds_workflow_start("data_cleaning", operation_data)
        
        # 3. 데이터 분석 단계
        sample_data = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'value': [10, None, 30, 40, 1000],  # 결측값과 이상값 포함
            'category': ['A', 'B', 'A', 'C', 'B']
        })
        
        data_summary = f"""데이터 요약:
- 행 수: {len(sample_data)}
- 열 수: {len(sample_data.columns)}
- 결측값: {sample_data.isnull().sum().sum()}개
- 숫자 열: {sample_data.select_dtypes(include=['number']).columns.tolist()}
- 범주 열: {sample_data.select_dtypes(include=['object']).columns.tolist()}

통계 요약:
{sample_data.describe().to_string()}
"""
        
        self.ai_ds_wrapper.trace_data_analysis_step(data_summary, "initial_inspection")
        time.sleep(0.5)  # 실제 처리 시간 시뮬레이션
        
        # 4. LLM 추천 단계
        recommendation_prompt = """데이터 정리 전문가로서 다음 데이터를 분석하고 정리 단계를 추천해주세요:

데이터 특성:
- 5행 3열의 데이터
- 'value' 열에 결측값 1개 발견
- 'value' 열에 이상값 가능성 (1000은 다른 값들에 비해 매우 큼)
- 'category' 열은 범주형 데이터

요청사항: 결측값과 이상값을 처리해주세요

단계별 추천사항을 제공해주세요."""

        recommendation_response = """# 데이터 정리 추천 단계

## 1. 데이터 품질 평가
- 결측값 패턴 분석: 'value' 열에 20% 결측률
- 이상값 탐지: 'value' 열에서 1000은 IQR 기준 이상값으로 판단

## 2. 결측값 처리 
- 'value' 열 결측값을 평균값(26.25)으로 대체
- 대안: 중앙값(25) 또는 최빈값 사용 가능

## 3. 이상값 처리
- IQR 방법으로 이상값 경계 계산: Q1=17.5, Q3=35, IQR=17.5
- 상한: Q3 + 1.5*IQR = 61.25 (1000은 이상값)
- 이상값을 상한값으로 클리핑 또는 제거 고려

## 4. 데이터 타입 최적화
- 'category' 열을 Category 타입으로 변환하여 메모리 효율성 증대

## 5. 검증
- 정리 후 데이터 품질 재검증
- 통계적 분포 확인"""

        self.ai_ds_wrapper.trace_llm_recommendation_step(
            recommendation_prompt, 
            recommendation_response, 
            "cleaning_strategy"
        )
        time.sleep(1.0)
        
        # 5. 코드 생성 단계
        code_generation_prompt = """앞서 추천한 단계를 바탕으로 데이터 정리 함수를 생성해주세요:

요구사항:
- 함수명: data_cleaner
- 결측값을 평균값으로 대체
- IQR 기준으로 이상값 처리
- 데이터 타입 최적화
- 정리된 데이터프레임 반환

Python 코드를 생성해주세요."""

        generated_code = '''def data_cleaner(data_raw):
    """
    데이터 정리 함수 - AI_DS_Team Enhanced Demo
    결측값 처리, 이상값 처리, 데이터 타입 최적화 수행
    """
    import pandas as pd
    import numpy as np
    
    # 입력 데이터가 딕셔너리인 경우 DataFrame으로 변환
    if isinstance(data_raw, dict):
        df = pd.DataFrame.from_dict(data_raw)
    else:
        df = data_raw.copy()
    
    print("🧹 데이터 정리 시작")
    print(f"원본 데이터 크기: {df.shape}")
    
    # 1. 결측값 처리
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().any():
            mean_value = df[col].mean()
            df[col].fillna(mean_value, inplace=True)
            print(f"'{col}' 열 결측값을 평균값 {mean_value:.2f}로 대체")
    
    # 2. 이상값 처리 (IQR 방법)
    for col in df.select_dtypes(include=[np.number]).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
        if outliers.any():
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
            print(f"'{col}' 열 이상값을 [{lower_bound:.2f}, {upper_bound:.2f}] 범위로 클리핑")
    
    # 3. 데이터 타입 최적화
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() / len(df) < 0.5:  # 카디널리티가 50% 미만인 경우
            df[col] = df[col].astype('category')
            print(f"'{col}' 열을 Category 타입으로 변환")
    
    print(f"정리된 데이터 크기: {df.shape}")
    print("✅ 데이터 정리 완료")
    
    return df'''

        self.ai_ds_wrapper.trace_code_generation_step(
            code_generation_prompt,
            generated_code,
            "data_cleaner_function"
        )
        time.sleep(1.5)
        
        # 6. 코드 실행 단계
        start_exec_time = time.time()
        try:
            # 실제 코드 실행 시뮬레이션
            exec(generated_code)
            
            # 생성된 함수 실행
            cleaned_data = locals()['data_cleaner'](sample_data)
            exec_time = time.time() - start_exec_time
            
            self.ai_ds_wrapper.trace_code_execution_step(
                generated_code,
                cleaned_data,
                exec_time
            )
            
            print(f"✅ 코드 실행 성공 (실행시간: {exec_time:.3f}초)")
            
        except Exception as e:
            exec_time = time.time() - start_exec_time
            error_msg = f"코드 실행 오류: {str(e)}"
            
            self.ai_ds_wrapper.trace_code_execution_step(
                generated_code,
                None,
                exec_time,
                error_msg
            )
            
            print(f"❌ 코드 실행 실패: {error_msg}")
            cleaned_data = sample_data  # 원본 데이터 사용
        
        # 7. 데이터 변환 추적
        self.ai_ds_wrapper.trace_data_transformation_step(
            sample_data,
            cleaned_data,
            "data_cleaning_transformation"
        )
        
        # 8. 워크플로우 완료
        workflow_summary = f"""# 데이터 정리 워크플로우 완료

## 처리 요약
- **요청**: 결측값과 이상값 처리
- **처리 단계**: {self.ai_ds_wrapper.step_counter}단계
- **소요 시간**: 약 4초

## 데이터 변화
- **원본**: {sample_data.shape[0]}행 {sample_data.shape[1]}열
- **정리 후**: {cleaned_data.shape[0]}행 {cleaned_data.shape[1]}열
- **결측값**: {sample_data.isnull().sum().sum()}개 → {cleaned_data.isnull().sum().sum()}개
- **데이터 타입 최적화**: 범주형 데이터 Category 타입 적용

## 생성된 아티팩트
- 데이터 분석 요약
- LLM 추천사항 (프롬프트 + 응답)
- 생성된 Python 코드
- 코드 실행 결과
- 변환 전후 데이터 샘플

## 품질 개선
- 결측값 0개로 개선
- 이상값 IQR 기준 정규화
- 메모리 사용량 최적화
"""

        self.ai_ds_wrapper.trace_workflow_completion(cleaned_data, workflow_summary)
        
        print("🎯 워크플로우 시뮬레이션 완료")
        return cleaned_data
    
    def demonstrate_nested_tracking(self):
        """중첩된 추적 구조 데모"""
        print("\n🔗 중첩된 추적 구조 데모")
        
        # 메인 작업
        main_span = self.ai_ds_wrapper.create_nested_span("Main Analysis Task")
        
        # 하위 작업들
        subtasks = [
            ("Data Validation", "데이터 유효성 검증"),
            ("Statistical Analysis", "통계 분석 수행"),
            ("Quality Assessment", "품질 평가")
        ]
        
        for subtask_name, subtask_desc in subtasks:
            subtask_span = self.ai_ds_wrapper.create_nested_span(
                subtask_name, 
                input_data={"description": subtask_desc}
            )
            
            # 시뮬레이션된 처리
            time.sleep(0.3)
            
            if subtask_span:
                subtask_span.end(
                    output={"status": "completed", "result": f"{subtask_desc} 완료"}
                )
            
            print(f"  ✅ {subtask_name} 완료")
        
        if main_span:
            main_span.end(output={"total_subtasks": len(subtasks), "status": "all_completed"})
        
        print("✅ 중첩된 추적 구조 데모 완료")
    
    def finalize_session(self):
        """세션 종료"""
        if self.ai_ds_wrapper:
            self.ai_ds_wrapper.finalize_agent_span(
                final_result="Enhanced tracking demo completed successfully",
                success=True
            )
        
        if self.session_tracer:
            self.session_tracer.end_user_session()
            print("✅ Langfuse 세션 종료")


async def main():
    """메인 실행 함수"""
    print("🔍 Enhanced Langfuse Tracking Test 시작")
    print("=" * 60)
    
    demo = EnhancedTrackingDemo()
    
    # Langfuse 초기화
    if not demo.initialize_langfuse():
        print("❌ Langfuse 초기화 실패. 테스트를 중단합니다.")
        return
    
    try:
        # 1. AI-Data-Science-Team 워크플로우 시뮬레이션
        cleaned_data = demo.simulate_ai_ds_team_workflow()
        
        # 2. 중첩된 추적 구조 데모
        demo.demonstrate_nested_tracking()
        
        # 결과 출력
        print("\n📊 처리 결과:")
        print("원본 데이터:")
        print(pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'value': [10, None, 30, 40, 1000],
            'category': ['A', 'B', 'A', 'C', 'B']
        }))
        
        print("\n정리된 데이터:")
        print(cleaned_data)
        
        print("\n🎯 Enhanced Tracking 테스트 완료!")
        print("\n🔍 Langfuse 대시보드에서 확인 가능한 정보:")
        print("   • 계층적 span 구조 (메인 → 워크플로우 → 개별 단계)")
        print("   • LLM 프롬프트/응답 아티팩트")
        print("   • 생성된 Python 코드 아티팩트")
        print("   • 코드 실행 결과 및 성능 메트릭")
        print("   • 데이터 변환 전후 샘플")
        print("   • 워크플로우 요약 (Markdown)")
        print("   • 중첩된 작업 구조")
        
        print(f"\n🌐 Langfuse 대시보드: http://mangugil.synology.me:3001")
        
    except Exception as e:
        print(f"❌ 테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 세션 정리
        demo.finalize_session()


if __name__ == "__main__":
    asyncio.run(main()) 