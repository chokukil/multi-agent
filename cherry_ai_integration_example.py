"""
Cherry AI에서 반도체 도메인 엔진 사용 예시
실제 통합 방법을 보여주는 코드
"""

import asyncio
from typing import Dict, Any
from services.semiconductor_domain_engine import analyze_semiconductor_data

class CherryAIWithSemiconductor:
    """반도체 도메인 엔진이 통합된 Cherry AI"""
    
    async def execute_analysis(self, user_query: str) -> Dict[str, Any]:
        """개선된 분석 실행 - 반도체 전문 분석 포함"""
        
        try:
            # 1. 기존 데이터 컨텍스트 준비
            data_context = {
                'data': st.session_state.current_data,
                'data_shape': st.session_state.current_data.shape,
                'columns': list(st.session_state.current_data.columns),
                'dtypes': st.session_state.current_data.dtypes.to_dict()
            }
            
            # 2. 🔬 반도체 도메인 엔진 우선 시도
            try:
                semiconductor_result = await analyze_semiconductor_data(
                    data=st.session_state.current_data,
                    user_query=user_query
                )
                
                # 반도체 도메인으로 높은 신뢰도로 판정된 경우
                confidence = semiconductor_result.get('context', {}).get('confidence_score', 0)
                
                if confidence > 0.7:  # 70% 이상 신뢰도
                    return self._format_semiconductor_analysis(semiconductor_result)
                    
            except Exception as e:
                print(f"반도체 분석 시도 중 오류: {e}")
                # 오류 시 일반 분석으로 fallback
            
            # 3. 일반 A2A 에이전트 분석으로 fallback
            return await self._general_agent_analysis(user_query, data_context)
            
        except Exception as e:
            return self._error_response(str(e))
    
    def _format_semiconductor_analysis(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """반도체 전문 분석 결과를 Cherry AI 형식으로 변환"""
        
        context = result.get('context', {})
        analysis = result.get('analysis', {})
        recommendations = result.get('recommendations', [])
        
        # Cherry AI 표준 형식으로 변환
        formatted_result = {
            'status': 'success',
            'summary': self._create_expert_summary(analysis),
            'artifacts': self._create_semiconductor_artifacts(analysis),
            'code': [],  # 반도체 분석은 주로 해석 중심
            'agent_contributions': {
                'semiconductor_expert': {
                    'summary': '반도체 제조 전문가 분석 완료',
                    'confidence': context.get('confidence_score', 0.9),
                    'process_type': context.get('process_type', 'unknown'),
                    'analysis_category': context.get('analysis_category', 'unknown')
                }
            },
            'execution_time': '실시간 분석',
            'selected_agents': ['semiconductor_domain_engine'],
            'domain_specific': True,
            'expert_recommendations': recommendations
        }
        
        return formatted_result
    
    def _create_expert_summary(self, analysis: Dict[str, Any]) -> str:
        """전문가 분석을 사용자 친화적 요약으로 변환"""
        
        process_interpretation = analysis.get('process_interpretation', '')
        technical_findings = analysis.get('technical_findings', [])
        quality_assessment = analysis.get('quality_assessment', {})
        
        summary = f"""🔬 **반도체 전문가 분석 완료**

**공정 해석:** {process_interpretation}

**주요 발견사항:**
"""
        
        for i, finding in enumerate(technical_findings[:3], 1):
            summary += f"\n{i}. {finding}"
        
        if quality_assessment:
            summary += f"""

**품질 평가:**
- 공정 능력: {quality_assessment.get('process_capability', 'N/A')}
- 수율 영향: {quality_assessment.get('yield_impact', 'N/A')}
- 스펙 준수: {quality_assessment.get('specification_compliance', 'N/A')}"""
        
        return summary
    
    def _create_semiconductor_artifacts(self, analysis: Dict[str, Any]) -> List[Dict]:
        """반도체 분석 결과를 시각화 아티팩트로 변환"""
        
        artifacts = []
        
        # 1. 품질 평가 테이블
        quality_assessment = analysis.get('quality_assessment', {})
        if quality_assessment:
            artifacts.append({
                'type': 'dataframe',
                'title': '품질 평가 요약',
                'data': pd.DataFrame([quality_assessment]).T.reset_index(),
                'description': '공정 능력 및 품질 지표 평가'
            })
        
        # 2. 개선 기회 리스트
        opportunities = analysis.get('optimization_opportunities', [])
        if opportunities:
            artifacts.append({
                'type': 'text',
                'title': '최적화 기회',
                'data': '\n'.join([f"• {opp}" for opp in opportunities]),
                'description': '확인된 공정 개선 기회들'
            })
        
        # 3. 리스크 지표
        risks = analysis.get('risk_indicators', [])
        if risks:
            artifacts.append({
                'type': 'text', 
                'title': '리스크 지표',
                'data': '\n'.join([f"⚠️ {risk}" for risk in risks]),
                'description': '주의 깊게 모니터링해야 할 리스크 요소들'
            })
        
        # 4. 실행 가능한 조치 방안
        actions = analysis.get('actionable_recommendations', [])
        if actions:
            artifacts.append({
                'type': 'text',
                'title': '즉시 실행 가능한 조치',
                'data': '\n'.join([f"🔧 {action}" for action in actions]),
                'description': '현장에서 바로 적용할 수 있는 구체적 조치 방안'
            })
        
        return artifacts

# 사용 예시
async def example_usage():
    """반도체 도메인 엔진 사용 예시"""
    
    # 1. 이온 주입 데이터로 가정
    sample_data = pd.DataFrame({
        'wafer_id': ['W001', 'W002', 'W003'] * 100,
        'x_position': range(300),
        'y_position': range(300), 
        'dose_measurement': [1.2e15, 1.1e15, 1.3e15] * 100,
        'energy_level': [25000, 25100, 24900] * 100,
        'beam_current': [5.2, 5.1, 5.3] * 100
    })
    
    # 2. 사용자 쿼리
    user_queries = [
        "도즈 균일성을 분석해주세요",
        "TW 값이 이상한데 원인을 찾아주세요", 
        "빔 전류 안정성을 확인해주세요",
        "이 공정 데이터에서 문제점을 찾아주세요"
    ]
    
    # 3. 각 쿼리별 분석 실행
    cherry_ai = CherryAIWithSemiconductor()
    
    for query in user_queries:
        print(f"\n📝 사용자 질문: {query}")
        print("=" * 50)
        
        # 실제로는 st.session_state.current_data에서 가져옴
        result = await analyze_semiconductor_data(sample_data, query)
        
        print(f"🔍 도메인 인식: {result['context']['process_type']}")
        print(f"📊 분석 카테고리: {result['context']['analysis_category']}")
        print(f"🎯 신뢰도: {result['context']['confidence_score']:.1%}")
        print(f"💡 전문 기법: {', '.join(result['context']['specialized_techniques'][:3])}")
        
        if result['context']['confidence_score'] > 0.7:
            print("✅ 반도체 전문 분석 수행됨")
        else:
            print("⚪ 일반 A2A 에이전트 분석으로 처리")

if __name__ == "__main__":
    # 테스트 실행
    asyncio.run(example_usage())