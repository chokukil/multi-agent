#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCP Tool: Report Writing Tools
보고서 작성 도구 - 문서 생성, 템플릿 관리, 형식 변환, 품질 검증
"""

import os
import sys
import json
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
import pandas as pd
import numpy as np
import uvicorn
import warnings
warnings.filterwarnings('ignore')

# FastMCP import
from mcp.server.fastmcp import FastMCP
from mcp.types import TextContent

# Get port from environment variable
SERVER_PORT = int(os.getenv('SERVER_PORT', '8019'))

# FastMCP 서버 생성
mcp = FastMCP("Report Writing Tools")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReportWritingTools:
    """보고서 작성 도구 클래스"""
    
    @staticmethod
    def generate_executive_summary(analysis_results: Dict[str, Any], 
                                 target_audience: str = "business",
                                 length: str = "medium",
                                 language: str = "korean") -> Dict[str, Any]:
        """실행 요약 생성"""
        try:
            # 핵심 인사이트 추출
            key_insights = ReportWritingTools._extract_key_insights(analysis_results)
            
            # 길이별 템플릿
            length_configs = {
                "short": {"max_words": 200, "key_points": 3},
                "medium": {"max_words": 500, "key_points": 5},
                "long": {"max_words": 800, "key_points": 7}
            }
            
            config = length_configs.get(length, length_configs["medium"])
            
            # 청중별 맞춤화
            audience_styles = {
                "business": {
                    "focus": "비즈니스 임팩트와 ROI",
                    "tone": "결정적이고 실행 중심적",
                    "keywords": ["효율성", "수익성", "성장", "기회", "위험"]
                },
                "technical": {
                    "focus": "방법론과 기술적 세부사항",
                    "tone": "정확하고 분석적",
                    "keywords": ["정확도", "성능", "알고리즘", "최적화", "검증"]
                },
                "executive": {
                    "focus": "전략적 시사점과 의사결정",
                    "tone": "간결하고 전략적",
                    "keywords": ["전략", "경쟁우위", "시장", "투자", "방향성"]
                },
                "general": {
                    "focus": "이해하기 쉬운 설명",
                    "tone": "명확하고 접근가능한",
                    "keywords": ["개선", "변화", "결과", "영향", "의미"]
                }
            }
            
            style = audience_styles.get(target_audience, audience_styles["business"])
            
            # 실행 요약 구조 생성
            summary_sections = {
                "overview": ReportWritingTools._create_overview_section(key_insights, style),
                "key_findings": ReportWritingTools._create_key_findings_section(
                    key_insights, config["key_points"], style
                ),
                "implications": ReportWritingTools._create_implications_section(key_insights, style),
                "recommendations": ReportWritingTools._create_recommendations_section(key_insights, style)
            }
            
            # 최종 실행 요약 조합
            if language == "korean":
                executive_summary = f"""
# 실행 요약

## 분석 개요
{summary_sections["overview"]}

## 주요 발견사항
{summary_sections["key_findings"]}

## 비즈니스 시사점
{summary_sections["implications"]}

## 핵심 권장사항
{summary_sections["recommendations"]}

---
*{datetime.now().strftime('%Y년 %m월 %d일')} 생성 | 대상: {target_audience} | 길이: {length}*
"""
            else:  # English
                executive_summary = f"""
# Executive Summary

## Analysis Overview
{summary_sections["overview"]}

## Key Findings
{summary_sections["key_findings"]}

## Business Implications
{summary_sections["implications"]}

## Key Recommendations
{summary_sections["recommendations"]}

---
*Generated on {datetime.now().strftime('%B %d, %Y')} | Audience: {target_audience} | Length: {length}*
"""
            
            # 품질 메트릭 계산
            word_count = len(executive_summary.split())
            quality_score = ReportWritingTools._calculate_summary_quality(
                executive_summary, config["max_words"], config["key_points"]
            )
            
            return {
                "executive_summary": executive_summary.strip(),
                "sections": summary_sections,
                "metadata": {
                    "target_audience": target_audience,
                    "length": length,
                    "language": language,
                    "word_count": word_count,
                    "target_word_count": config["max_words"],
                    "key_points_count": config["key_points"],
                    "quality_score": quality_score,
                    "style_focus": style["focus"]
                },
                "interpretation": f"{target_audience} 대상 {length} 길이 실행 요약 생성 완료"
            }
            
        except Exception as e:
            return {"error": f"실행 요약 생성 중 오류: {str(e)}"}
    
    @staticmethod
    def _extract_key_insights(analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """분석 결과에서 핵심 인사이트 추출"""
        insights = {
            "statistics": [],
            "patterns": [],
            "anomalies": [],
            "trends": [],
            "recommendations": [],
            "metrics": {}
        }
        
        # 통계 분석 결과에서 인사이트 추출
        if "statistical_results" in analysis_results:
            stats = analysis_results["statistical_results"]
            
            for var, result in stats.items():
                if isinstance(result, dict):
                    if "descriptive_stats" in result:
                        desc_stats = result["descriptive_stats"]
                        if "mean" in desc_stats and "std" in desc_stats:
                            cv = desc_stats["std"] / desc_stats["mean"] if desc_stats["mean"] != 0 else 0
                            if cv > 0.5:
                                insights["patterns"].append(f"{var} 변수에서 높은 변동성 발견 (CV: {cv:.2f})")
                            
                            if "skewness" in desc_stats and abs(desc_stats["skewness"]) > 1:
                                insights["patterns"].append(f"{var} 변수에서 비대칭 분포 발견")
        
        # ML 결과에서 인사이트 추출
        if "model_summary" in analysis_results:
            model_info = analysis_results["model_summary"]
            if "r_squared" in model_info:
                r2 = model_info["r_squared"]
                if r2 > 0.8:
                    insights["statistics"].append(f"높은 예측 정확도 달성 (R² = {r2:.3f})")
                elif r2 < 0.3:
                    insights["statistics"].append(f"낮은 예측 성능 (R² = {r2:.3f}) - 추가 특성 필요")
        
        # 시계열 분석에서 트렌드 추출
        if "trend_analysis" in analysis_results:
            trend_info = analysis_results["trend_analysis"]
            if "trend_direction" in trend_info:
                direction = trend_info["trend_direction"]
                insights["trends"].append(f"데이터에서 {direction} 트렌드 확인")
        
        # 이상치 정보 추출
        if "outlier_info" in analysis_results:
            outlier_info = analysis_results["outlier_info"]
            total_outliers = sum([info.get("outlier_count", 0) for info in outlier_info.values()])
            if total_outliers > 0:
                insights["anomalies"].append(f"총 {total_outliers}개의 이상치 발견")
        
        # 메트릭 정보 추출
        for key, value in analysis_results.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                insights["metrics"][key] = value
        
        return insights
    
    @staticmethod
    def _create_overview_section(insights: Dict[str, Any], style: Dict[str, str]) -> str:
        """개요 섹션 생성"""
        
        overview_parts = []
        
        # 분석 범위
        total_metrics = len(insights["metrics"])
        total_findings = len(insights["statistics"]) + len(insights["patterns"])
        
        overview_parts.append(f"이 분석은 {total_metrics}개의 주요 지표를 대상으로 수행되었으며, "
                            f"{total_findings}개의 핵심 발견사항을 도출했습니다.")
        
        # 스타일에 따른 포커스
        if "비즈니스" in style["focus"]:
            overview_parts.append("분석 결과는 운영 효율성 개선과 수익성 향상을 위한 "
                               "실행 가능한 인사이트를 제공합니다.")
        elif "기술적" in style["focus"]:
            overview_parts.append("통계적으로 검증된 방법론을 통해 데이터의 패턴과 "
                               "특성을 과학적으로 분석했습니다.")
        elif "전략적" in style["focus"]:
            overview_parts.append("전략적 의사결정을 지원하는 핵심 인사이트와 "
                               "시장 기회를 식별했습니다.")
        
        return " ".join(overview_parts)
    
    @staticmethod
    def _create_key_findings_section(insights: Dict[str, Any], 
                                   max_points: int, 
                                   style: Dict[str, str]) -> str:
        """주요 발견사항 섹션 생성"""
        
        all_findings = []
        all_findings.extend(insights["statistics"])
        all_findings.extend(insights["patterns"])
        all_findings.extend(insights["trends"])
        all_findings.extend(insights["anomalies"][:2])  # 이상치는 최대 2개만
        
        # 중요도순 정렬 (길이와 키워드 기반 간단한 평가)
        scored_findings = []
        for finding in all_findings:
            score = 0
            for keyword in style["keywords"]:
                if keyword in finding.lower():
                    score += 2
            if any(char.isdigit() for char in finding):  # 수치 포함시 가점
                score += 1
            scored_findings.append((score, finding))
        
        scored_findings.sort(reverse=True)
        top_findings = [finding for score, finding in scored_findings[:max_points]]
        
        # 번호 매기기
        numbered_findings = []
        for i, finding in enumerate(top_findings, 1):
            numbered_findings.append(f"{i}. {finding}")
        
        return "\n".join(numbered_findings)
    
    @staticmethod
    def _create_implications_section(insights: Dict[str, Any], style: Dict[str, str]) -> str:
        """시사점 섹션 생성"""
        
        implications = []
        
        # 패턴 기반 시사점
        if insights["patterns"]:
            implications.append("• 발견된 데이터 패턴은 기존 가정을 재검토할 필요성을 시사합니다.")
        
        # 트렌드 기반 시사점
        if insights["trends"]:
            implications.append("• 확인된 트렌드는 중장기 전략 수립에 중요한 고려사항입니다.")
        
        # 이상치 기반 시사점
        if insights["anomalies"]:
            implications.append("• 이상치 분석 결과는 품질 관리 체계의 강화가 필요함을 보여줍니다.")
        
        # 스타일별 맞춤 시사점
        if "비즈니스" in style["focus"]:
            implications.append("• 분석 결과는 운영 프로세스 최적화를 통한 비용 절감 기회를 제시합니다.")
        elif "기술적" in style["focus"]:
            implications.append("• 데이터 품질과 분석 방법론의 개선이 예측 정확도를 높일 것입니다.")
        
        return "\n".join(implications)
    
    @staticmethod
    def _create_recommendations_section(insights: Dict[str, Any], style: Dict[str, str]) -> str:
        """권장사항 섹션 생성"""
        
        recommendations = []
        
        # 기본 권장사항
        if insights["statistics"]:
            recommendations.append("1. 핵심 지표에 대한 정기적 모니터링 체계 구축")
        
        if insights["patterns"]:
            recommendations.append("2. 발견된 패턴을 활용한 예측 모델 개발")
        
        if insights["anomalies"]:
            recommendations.append("3. 이상치 탐지 시스템의 도입 및 운영")
        
        # 스타일별 특화 권장사항
        if "비즈니스" in style["focus"]:
            recommendations.append("4. 분석 결과를 반영한 비즈니스 프로세스 개선")
            recommendations.append("5. ROI 기반 우선순위로 개선 과제 추진")
        elif "기술적" in style["focus"]:
            recommendations.append("4. 고급 분석 기법 도입을 통한 인사이트 확장")
            recommendations.append("5. 데이터 파이프라인 자동화 및 품질 관리")
        elif "전략적" in style["focus"]:
            recommendations.append("4. 경쟁우위 확보를 위한 데이터 기반 전략 수립")
            recommendations.append("5. 데이터 거버넌스 체계 구축")
        
        return "\n".join(recommendations)
    
    @staticmethod
    def _calculate_summary_quality(summary: str, target_words: int, key_points: int) -> float:
        """실행 요약 품질 점수 계산"""
        
        word_count = len(summary.split())
        
        # 길이 점수 (목표 대비 ±20% 이내면 만점)
        length_ratio = word_count / target_words
        if 0.8 <= length_ratio <= 1.2:
            length_score = 100
        elif 0.6 <= length_ratio <= 1.4:
            length_score = 80
        else:
            length_score = 60
        
        # 구조 점수 (헤딩, 번호 매기기 등)
        structure_indicators = ["##", "#", "1.", "2.", "3.", "•", "-"]
        structure_count = sum(1 for indicator in structure_indicators if indicator in summary)
        structure_score = min(structure_count * 15, 100)
        
        # 내용 점수 (수치, 구체성)
        numeric_content = len([word for word in summary.split() if any(char.isdigit() for char in word)])
        content_score = min(numeric_content * 5, 100)
        
        # 가독성 점수 (문장 길이, 전문용어 비율)
        sentences = summary.split('.')
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
        if 10 <= avg_sentence_length <= 20:
            readability_score = 100
        elif 8 <= avg_sentence_length <= 25:
            readability_score = 80
        else:
            readability_score = 60
        
        # 전체 점수 (가중평균)
        overall_score = (
            length_score * 0.25 +
            structure_score * 0.25 +
            content_score * 0.25 +
            readability_score * 0.25
        )
        
        return round(overall_score, 1)
    
    @staticmethod
    def create_report_template(report_type: str = "technical",
                             sections: List[str] = None,
                             language: str = "korean") -> Dict[str, Any]:
        """보고서 템플릿 생성"""
        try:
            # 기본 섹션 정의
            default_sections = {
                "technical": [
                    "executive_summary", "introduction", "methodology", 
                    "results", "discussion", "conclusions", "recommendations", "appendix"
                ],
                "business": [
                    "executive_summary", "background", "key_findings", 
                    "business_impact", "recommendations", "next_steps"
                ],
                "research": [
                    "abstract", "introduction", "literature_review", "methodology",
                    "results", "discussion", "conclusions", "references"
                ],
                "presentation": [
                    "title_slide", "agenda", "key_findings", "insights",
                    "recommendations", "next_steps", "questions"
                ]
            }
            
            if sections is None:
                sections = default_sections.get(report_type, default_sections["technical"])
            
            # 언어별 섹션 제목
            korean_titles = {
                "executive_summary": "실행 요약",
                "abstract": "초록",
                "introduction": "서론",
                "background": "배경",
                "literature_review": "문헌 검토",
                "methodology": "방법론",
                "results": "결과",
                "key_findings": "주요 발견사항",
                "discussion": "논의",
                "business_impact": "비즈니스 임팩트",
                "insights": "인사이트",
                "conclusions": "결론",
                "recommendations": "권장사항",
                "next_steps": "다음 단계",
                "appendix": "부록",
                "references": "참고문헌",
                "title_slide": "제목",
                "agenda": "목차",
                "questions": "질의응답"
            }
            
            english_titles = {
                "executive_summary": "Executive Summary",
                "abstract": "Abstract",
                "introduction": "Introduction",
                "background": "Background",
                "literature_review": "Literature Review",
                "methodology": "Methodology",
                "results": "Results",
                "key_findings": "Key Findings",
                "discussion": "Discussion",
                "business_impact": "Business Impact",
                "insights": "Insights",
                "conclusions": "Conclusions",
                "recommendations": "Recommendations",
                "next_steps": "Next Steps",
                "appendix": "Appendix",
                "references": "References",
                "title_slide": "Title",
                "agenda": "Agenda",
                "questions": "Q&A"
            }
            
            titles = korean_titles if language == "korean" else english_titles
            
            # 템플릿 생성
            template_structure = []
            section_details = {}
            
            for section in sections:
                title = titles.get(section, section.replace("_", " ").title())
                template_structure.append({
                    "section_id": section,
                    "title": title,
                    "order": len(template_structure) + 1
                })
                
                # 각 섹션별 가이드라인
                section_details[section] = ReportWritingTools._get_section_guidelines(section, language)
            
            # 템플릿 문서 생성
            template_content = ReportWritingTools._generate_template_content(
                template_structure, section_details, report_type, language
            )
            
            return {
                "template_content": template_content,
                "template_structure": template_structure,
                "section_details": section_details,
                "metadata": {
                    "report_type": report_type,
                    "language": language,
                    "sections_count": len(sections),
                    "estimated_pages": len(sections) * 2,  # 섹션당 평균 2페이지
                    "created_date": datetime.now().isoformat()
                },
                "writing_guidelines": ReportWritingTools._get_writing_guidelines(report_type, language),
                "interpretation": f"{report_type} 보고서 템플릿 생성 완료 ({len(sections)}개 섹션)"
            }
            
        except Exception as e:
            return {"error": f"보고서 템플릿 생성 중 오류: {str(e)}"}
    
    @staticmethod
    def _get_section_guidelines(section: str, language: str) -> Dict[str, Any]:
        """섹션별 작성 가이드라인"""
        
        guidelines = {
            "executive_summary": {
                "purpose": "핵심 내용 요약 및 주요 권장사항 제시",
                "length": "1-2페이지",
                "key_elements": ["목적", "주요 발견사항", "권장사항", "기대효과"],
                "writing_tips": ["간결하고 명확한 표현", "수치 기반 근거", "실행 가능한 권장사항"]
            },
            "methodology": {
                "purpose": "분석 방법과 절차의 체계적 설명",
                "length": "2-3페이지",
                "key_elements": ["데이터 수집", "분석 방법", "도구 및 기법", "검증 방법"],
                "writing_tips": ["재현 가능한 수준의 상세함", "가정과 제약사항 명시", "선택 근거 제시"]
            },
            "results": {
                "purpose": "분석 결과의 객관적 제시",
                "length": "3-5페이지",
                "key_elements": ["주요 결과", "통계적 유의성", "시각화", "추가 분석"],
                "writing_tips": ["사실 중심 서술", "해석과 결과 구분", "시각적 보조자료 활용"]
            },
            "recommendations": {
                "purpose": "실행 가능한 권장사항과 실행 계획",
                "length": "1-2페이지",
                "key_elements": ["핵심 권장사항", "우선순위", "실행 방안", "기대 효과"],
                "writing_tips": ["구체적 액션 아이템", "담당자/일정 명시", "ROI 산정"]
            }
        }
        
        return guidelines.get(section, {
            "purpose": "섹션 내용 작성",
            "length": "1-2페이지",
            "key_elements": ["주요 내용"],
            "writing_tips": ["명확하고 간결한 작성"]
        })
    
    @staticmethod
    def _generate_template_content(structure: List[Dict], 
                                 section_details: Dict, 
                                 report_type: str, 
                                 language: str) -> str:
        """템플릿 내용 생성"""
        
        if language == "korean":
            title = f"# {report_type.upper()} 보고서\n\n"
            meta = f"**작성일:** {datetime.now().strftime('%Y년 %m월 %d일')}\n"
            meta += f"**보고서 유형:** {report_type}\n\n"
            
            toc = "## 목차\n\n"
            for item in structure:
                toc += f"{item['order']}. [{item['title']}](#{item['section_id']})\n"
            toc += "\n---\n\n"
            
        else:
            title = f"# {report_type.upper()} Report\n\n"
            meta = f"**Date:** {datetime.now().strftime('%B %d, %Y')}\n"
            meta += f"**Report Type:** {report_type}\n\n"
            
            toc = "## Table of Contents\n\n"
            for item in structure:
                toc += f"{item['order']}. [{item['title']}](#{item['section_id']})\n"
            toc += "\n---\n\n"
        
        # 각 섹션 템플릿
        sections_content = ""
        for item in structure:
            section_id = item['section_id']
            title = item['title']
            details = section_details.get(section_id, {})
            
            sections_content += f"## {title}\n\n"
            sections_content += f"**목적:** {details.get('purpose', 'N/A')}\n\n"
            sections_content += f"**예상 길이:** {details.get('length', '1-2페이지')}\n\n"
            
            if 'key_elements' in details:
                sections_content += "**포함할 주요 요소:**\n"
                for element in details['key_elements']:
                    sections_content += f"- {element}\n"
                sections_content += "\n"
            
            if 'writing_tips' in details:
                sections_content += "**작성 팁:**\n"
                for tip in details['writing_tips']:
                    sections_content += f"- {tip}\n"
                sections_content += "\n"
            
            sections_content += "*[여기에 내용을 작성하세요]*\n\n"
            sections_content += "---\n\n"
        
        return title + meta + toc + sections_content
    
    @staticmethod
    def _get_writing_guidelines(report_type: str, language: str) -> Dict[str, List[str]]:
        """보고서 유형별 작성 가이드라인"""
        
        guidelines = {
            "general": [
                "명확하고 간결한 문장 사용",
                "논리적이고 체계적인 구성",
                "데이터와 근거 기반 서술",
                "독자 관점에서 이해하기 쉽게 작성"
            ],
            "technical": [
                "기술적 정확성 확보",
                "재현 가능한 수준의 상세함",
                "전문용어 사용시 정의 제공",
                "그래프와 표를 적극 활용"
            ],
            "business": [
                "비즈니스 임팩트 강조",
                "실행 가능한 권장사항 제시",
                "ROI와 비용효과 분석 포함",
                "의사결정자 관점에서 작성"
            ],
            "research": [
                "학술적 엄밀성 유지",
                "선행 연구와의 비교 분석",
                "연구 한계점 명시",
                "인용과 참고문헌 정확히 기재"
            ]
        }
        
        return {
            "general_guidelines": guidelines["general"],
            "specific_guidelines": guidelines.get(report_type, guidelines["general"])
        }
    
    @staticmethod
    def format_report_content(content: str,
                            output_format: str = "markdown",
                            styling_options: Dict[str, Any] = None) -> Dict[str, Any]:
        """보고서 내용 형식 변환"""
        try:
            if styling_options is None:
                styling_options = {}
            
            formatted_content = ""
            conversion_info = {}
            
            if output_format.lower() == "html":
                formatted_content = ReportWritingTools._convert_to_html(content, styling_options)
                conversion_info["format"] = "HTML"
                conversion_info["features"] = ["CSS 스타일링", "반응형 디자인", "내비게이션"]
                
            elif output_format.lower() == "latex":
                formatted_content = ReportWritingTools._convert_to_latex(content, styling_options)
                conversion_info["format"] = "LaTeX"
                conversion_info["features"] = ["PDF 출력", "학술 포맷", "수식 지원"]
                
            elif output_format.lower() == "docx":
                # Word 문서 형식 (간단한 변환)
                formatted_content = ReportWritingTools._convert_to_docx_format(content)
                conversion_info["format"] = "Word Document"
                conversion_info["features"] = ["편집 가능", "추적 변경", "댓글 기능"]
                
            elif output_format.lower() == "json":
                formatted_content = ReportWritingTools._convert_to_json(content)
                conversion_info["format"] = "JSON"
                conversion_info["features"] = ["구조화된 데이터", "API 호환", "자동 처리"]
                
            else:  # markdown (기본값)
                formatted_content = content  # 이미 마크다운 형식
                conversion_info["format"] = "Markdown"
                conversion_info["features"] = ["GitHub 호환", "간단한 문법", "버전 관리"]
            
            # 포맷 검증
            validation_result = ReportWritingTools._validate_format(formatted_content, output_format)
            
            return {
                "formatted_content": formatted_content,
                "conversion_info": conversion_info,
                "validation": validation_result,
                "metadata": {
                    "input_format": "markdown",
                    "output_format": output_format,
                    "content_length": len(formatted_content),
                    "conversion_date": datetime.now().isoformat(),
                    "styling_applied": bool(styling_options)
                },
                "interpretation": f"보고서를 {output_format} 형식으로 변환 완료"
            }
            
        except Exception as e:
            return {"error": f"형식 변환 중 오류: {str(e)}"}
    
    @staticmethod
    def _convert_to_html(content: str, styling: Dict[str, Any]) -> str:
        """마크다운을 HTML로 변환"""
        
        # 기본 CSS 스타일
        default_style = """
        <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 40px; line-height: 1.6; }
        h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
        h2 { color: #34495e; margin-top: 30px; }
        h3 { color: #7f8c8d; }
        .meta { color: #7f8c8d; font-style: italic; }
        .highlight { background-color: #f1c40f; padding: 2px 4px; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
        th { background-color: #f2f2f2; }
        </style>
        """
        
        custom_style = styling.get("custom_css", "")
        
        # 간단한 마크다운-HTML 변환
        html_content = content
        
        # 헤더 변환
        html_content = html_content.replace("# ", "<h1>").replace("\n", "</h1>\n", 1)
        html_content = html_content.replace("## ", "<h2>").replace("\n", "</h2>\n")
        html_content = html_content.replace("### ", "<h3>").replace("\n", "</h3>\n")
        
        # 리스트 변환 (간단한 버전)
        lines = html_content.split('\n')
        in_list = False
        processed_lines = []
        
        for line in lines:
            if line.strip().startswith('- ') or line.strip().startswith('* '):
                if not in_list:
                    processed_lines.append('<ul>')
                    in_list = True
                processed_lines.append(f'<li>{line.strip()[2:]}</li>')
            else:
                if in_list:
                    processed_lines.append('</ul>')
                    in_list = False
                processed_lines.append(line)
        
        if in_list:
            processed_lines.append('</ul>')
        
        html_content = '\n'.join(processed_lines)
        
        # 굵은 글씨와 기울임꼴
        html_content = html_content.replace('**', '<strong>', 1).replace('**', '</strong>', 1)
        html_content = html_content.replace('*', '<em>', 1).replace('*', '</em>', 1)
        
        # 단락 생성
        html_content = html_content.replace('\n\n', '</p><p>')
        html_content = f'<p>{html_content}</p>'
        
        # 전체 HTML 문서 구성
        full_html = f"""
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>분석 보고서</title>
    {default_style}
    {custom_style}
</head>
<body>
    {html_content}
</body>
</html>
"""
        
        return full_html
    
    @staticmethod
    def _convert_to_latex(content: str, styling: Dict[str, Any]) -> str:
        """마크다운을 LaTeX로 변환"""
        
        # LaTeX 문서 헤더
        latex_header = r"""
\documentclass[12pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[korean]{babel}
\usepackage{kotex}
\usepackage{geometry}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{hyperref}

\geometry{margin=2.5cm}
\setlength{\parindent}{0pt}
\setlength{\parskip}{6pt}

\title{데이터 분석 보고서}
\author{CherryAI}
\date{\today}

\begin{document}
\maketitle
\tableofcontents
\newpage

"""
        
        # 간단한 마크다운-LaTeX 변환
        latex_content = content
        
        # 헤더 변환
        latex_content = latex_content.replace("# ", r"\section{").replace("\n", "}\n", 1)
        latex_content = latex_content.replace("## ", r"\subsection{").replace("\n", "}\n")
        latex_content = latex_content.replace("### ", r"\subsubsection{").replace("\n", "}\n")
        
        # 굵은 글씨와 기울임꼴
        latex_content = latex_content.replace('**', r'\textbf{', 1).replace('**', '}', 1)
        latex_content = latex_content.replace('*', r'\textit{', 1).replace('*', '}', 1)
        
        # 리스트 변환
        lines = latex_content.split('\n')
        processed_lines = []
        in_list = False
        
        for line in lines:
            if line.strip().startswith('- ') or line.strip().startswith('* '):
                if not in_list:
                    processed_lines.append(r'\begin{itemize}')
                    in_list = True
                processed_lines.append(f'\\item {line.strip()[2:]}')
            else:
                if in_list:
                    processed_lines.append(r'\end{itemize}')
                    in_list = False
                processed_lines.append(line)
        
        if in_list:
            processed_lines.append(r'\end{itemize}')
        
        latex_content = '\n'.join(processed_lines)
        
        # LaTeX 문서 마무리
        latex_footer = r"""

\end{document}
"""
        
        return latex_header + latex_content + latex_footer
    
    @staticmethod
    def _convert_to_docx_format(content: str) -> str:
        """Word 문서 형식용 변환 (마크다운 유지하되 Word 호환성 개선)"""
        
        # Word에서 잘 읽히는 형태로 조정
        docx_content = content
        
        # 특수 문자 정리
        docx_content = docx_content.replace('—', '-')
        docx_content = docx_content.replace('"', '"').replace('"', '"')
        docx_content = docx_content.replace(''', "'").replace(''', "'")
        
        # Word용 메타데이터 추가
        word_meta = f"""
문서 정보:
- 생성일: {datetime.now().strftime('%Y년 %m월 %d일')}
- 형식: Microsoft Word 호환
- 편집: 추적 변경 활성화 권장

---

"""
        
        return word_meta + docx_content
    
    @staticmethod
    def _convert_to_json(content: str) -> str:
        """JSON 형식으로 변환"""
        
        # 간단한 구조화
        lines = content.split('\n')
        sections = []
        current_section = None
        
        for line in lines:
            line = line.strip()
            if line.startswith('# '):
                if current_section:
                    sections.append(current_section)
                current_section = {
                    "type": "title",
                    "level": 1,
                    "content": line[2:],
                    "subsections": []
                }
            elif line.startswith('## '):
                if current_section:
                    current_section["subsections"].append({
                        "type": "heading",
                        "level": 2,
                        "content": line[3:],
                        "text": []
                    })
            elif line and current_section and current_section["subsections"]:
                current_section["subsections"][-1]["text"].append(line)
            elif line and current_section:
                if "text" not in current_section:
                    current_section["text"] = []
                current_section["text"].append(line)
        
        if current_section:
            sections.append(current_section)
        
        json_structure = {
            "document_type": "analysis_report",
            "created_date": datetime.now().isoformat(),
            "sections": sections,
            "metadata": {
                "total_sections": len(sections),
                "format": "structured_json"
            }
        }
        
        return json.dumps(json_structure, indent=2, ensure_ascii=False)
    
    @staticmethod
    def _validate_format(content: str, format_type: str) -> Dict[str, Any]:
        """형식 검증"""
        
        validation = {
            "is_valid": True,
            "issues": [],
            "suggestions": []
        }
        
        if format_type == "html":
            # HTML 기본 검증
            if not content.startswith("<!DOCTYPE"):
                validation["issues"].append("DOCTYPE 선언 누락")
            if "<title>" not in content:
                validation["suggestions"].append("title 태그 추가 권장")
                
        elif format_type == "latex":
            # LaTeX 기본 검증
            if "\\begin{document}" not in content:
                validation["issues"].append("document 환경 누락")
                validation["is_valid"] = False
            if "\\end{document}" not in content:
                validation["issues"].append("document 종료 누락")
                validation["is_valid"] = False
                
        elif format_type == "json":
            # JSON 유효성 검증
            try:
                json.loads(content)
            except json.JSONDecodeError as e:
                validation["issues"].append(f"JSON 형식 오류: {str(e)}")
                validation["is_valid"] = False
        
        # 공통 검증
        if len(content) < 100:
            validation["suggestions"].append("내용이 너무 짧습니다")
        
        return validation


# MCP 도구 등록
@mcp.tool("generate_executive_summary")
def generate_executive_summary(analysis_results: Dict[str, Any], 
                             target_audience: str = "business",
                             length: str = "medium",
                             language: str = "korean") -> Dict[str, Any]:
    """
    분석 결과를 바탕으로 실행 요약을 생성합니다.
    
    Args:
        analysis_results: 분석 결과 데이터
        target_audience: 대상 청중 (business, technical, executive, general)
        length: 요약 길이 (short, medium, long)
        language: 언어 (korean, english)
    
    Returns:
        실행 요약 결과
    """
    return ReportWritingTools.generate_executive_summary(analysis_results, target_audience, length, language)


@mcp.tool("create_report_template")
def create_report_template(report_type: str = "technical",
                         sections: List[str] = None,
                         language: str = "korean") -> Dict[str, Any]:
    """
    보고서 템플릿을 생성합니다.
    
    Args:
        report_type: 보고서 유형 (technical, business, research, presentation)
        sections: 포함할 섹션 리스트
        language: 언어 (korean, english)
    
    Returns:
        보고서 템플릿
    """
    return ReportWritingTools.create_report_template(report_type, sections, language)


@mcp.tool("format_report_content")
def format_report_content(content: str,
                        output_format: str = "markdown",
                        styling_options: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    보고서 내용을 다른 형식으로 변환합니다.
    
    Args:
        content: 변환할 보고서 내용 (마크다운 형식)
        output_format: 출력 형식 (html, latex, docx, json, markdown)
        styling_options: 스타일링 옵션
    
    Returns:
        변환된 보고서
    """
    return ReportWritingTools.format_report_content(content, output_format, styling_options)


@mcp.tool("analyze_report_quality")
def analyze_report_quality(report_content: str,
                         report_type: str = "technical",
                         target_metrics: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    보고서 품질을 분석하고 개선 제안을 제공합니다.
    
    Args:
        report_content: 분석할 보고서 내용
        report_type: 보고서 유형
        target_metrics: 목표 지표
    
    Returns:
        품질 분석 결과
    """
    try:
        if target_metrics is None:
            target_metrics = {"min_words": 500, "max_words": 5000}
        
        # 기본 메트릭 계산
        word_count = len(report_content.split())
        char_count = len(report_content)
        paragraph_count = len([p for p in report_content.split('\n\n') if p.strip()])
        
        # 구조 분석
        headers = len([line for line in report_content.split('\n') if line.strip().startswith('#')])
        lists = len([line for line in report_content.split('\n') if line.strip().startswith(('-', '*', '1.'))])
        
        # 품질 점수 계산
        quality_scores = {}
        
        # 길이 점수
        min_words = target_metrics.get("min_words", 500)
        max_words = target_metrics.get("max_words", 5000)
        if min_words <= word_count <= max_words:
            quality_scores["length"] = 100
        elif word_count < min_words:
            quality_scores["length"] = (word_count / min_words) * 100
        else:
            quality_scores["length"] = max(0, 100 - ((word_count - max_words) / max_words) * 50)
        
        # 구조 점수
        structure_score = min(headers * 20 + lists * 5, 100)
        quality_scores["structure"] = structure_score
        
        # 가독성 점수 (평균 문장 길이 기반)
        sentences = [s for s in report_content.replace('\n', ' ').split('.') if s.strip()]
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
        
        if 10 <= avg_sentence_length <= 20:
            quality_scores["readability"] = 100
        elif 5 <= avg_sentence_length <= 30:
            quality_scores["readability"] = 80
        else:
            quality_scores["readability"] = 60
        
        # 내용 풍부도 (수치, 구체성)
        numeric_content = len([word for word in report_content.split() if any(char.isdigit() for char in word)])
        content_richness = min(numeric_content * 2, 100)
        quality_scores["content_richness"] = content_richness
        
        # 전체 점수
        overall_score = sum(quality_scores.values()) / len(quality_scores)
        
        # 개선 제안
        suggestions = []
        if quality_scores["length"] < 70:
            if word_count < min_words:
                suggestions.append(f"내용을 더 추가하세요 (현재: {word_count}단어, 목표: {min_words}+단어)")
            else:
                suggestions.append(f"내용을 간소화하세요 (현재: {word_count}단어, 권장: {max_words}단어 이하)")
        
        if quality_scores["structure"] < 60:
            suggestions.append("헤딩과 리스트를 추가하여 구조를 개선하세요")
        
        if quality_scores["readability"] < 70:
            suggestions.append("문장 길이를 조정하여 가독성을 개선하세요")
        
        if quality_scores["content_richness"] < 50:
            suggestions.append("구체적인 수치와 데이터를 더 포함하세요")
        
        return {
            "quality_metrics": {
                "word_count": word_count,
                "character_count": char_count,
                "paragraph_count": paragraph_count,
                "header_count": headers,
                "list_count": lists,
                "average_sentence_length": round(avg_sentence_length, 1)
            },
            "quality_scores": quality_scores,
            "overall_score": round(overall_score, 1),
            "grade": ReportWritingTools._get_quality_grade(overall_score),
            "suggestions": suggestions,
            "target_metrics": target_metrics,
            "analysis_summary": {
                "strengths": [key for key, score in quality_scores.items() if score >= 80],
                "improvements_needed": [key for key, score in quality_scores.items() if score < 60]
            },
            "interpretation": f"보고서 품질 점수: {overall_score:.1f}/100 ({ReportWritingTools._get_quality_grade(overall_score)})"
        }
        
    except Exception as e:
        return {"error": f"보고서 품질 분석 중 오류: {str(e)}"}


def _get_quality_grade(score: float) -> str:
    """품질 점수를 등급으로 변환"""
    if score >= 90:
        return "A (우수)"
    elif score >= 80:
        return "B (양호)"
    elif score >= 70:
        return "C (보통)"
    elif score >= 60:
        return "D (미흡)"
    else:
        return "F (개선 필요)"


ReportWritingTools._get_quality_grade = _get_quality_grade


if __name__ == "__main__":
    import sys
    import logging
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting Report Writing Tools MCP server on port {SERVER_PORT}...")
    
    try:
        # Get the SSE app and run it on the specified port
        app = mcp.sse_app()
        uvicorn.run(app, host="0.0.0.0", port=SERVER_PORT)
    except Exception as e:
        logger.error(f"Failed to start MCP server: {e}")
        sys.exit(1)