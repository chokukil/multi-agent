"""
Smart Download Manager - Context-Aware Enhanced Formats

Two-tier download system:
1. Raw artifacts: Direct A2A agent outputs (Chart JSON, Table CSV, Code PY, Image PNG)
2. Enhanced formats: Context-aware additional formats (PDF reports, Jupyter notebooks)
"""

import streamlit as st
import pandas as pd
import json
import io
import zipfile
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from pathlib import Path
import base64

from ..models import EnhancedArtifact

logger = logging.getLogger(__name__)


class SmartDownloadManager:
    """Smart download manager with context-aware formatting"""
    
    def __init__(self):
        """Initialize download manager"""
        self.raw_formats = {
            'plotly_chart': 'json',
            'table': 'csv', 
            'code': 'py',
            'image': 'png',
            'markdown': 'md'
        }
        
        self.enhanced_formats = {
            'business_user': ['pdf', 'xlsx', 'pptx'],
            'developer': ['ipynb', 'html', 'json'],
            'analyst': ['xlsx', 'pdf', 'html'],
            'researcher': ['pdf', 'latex', 'bib']
        }
        
    def generate_download_options(self,
                                 artifacts: List[EnhancedArtifact],
                                 user_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate download options for artifacts collection
        
        Returns:
            Dict with various download options
        """
        try:
            download_options = {
                'individual_raw': [],
                'individual_enhanced': [], 
                'bulk_raw': None,
                'bulk_enhanced': [],
                'combined_report': None
            }
            
            # Individual raw downloads (always available)
            for artifact in artifacts:
                raw_option = self._create_raw_download(artifact)
                if raw_option:
                    download_options['individual_raw'].append(raw_option)
            
            # Individual enhanced downloads
            user_type = self._determine_user_type(user_context)
            for artifact in artifacts:
                enhanced_options = self._create_enhanced_downloads(artifact, user_type)
                download_options['individual_enhanced'].extend(enhanced_options)
            
            # Bulk raw download (ZIP)
            bulk_zip = self._create_bulk_zip(artifacts)
            if bulk_zip:
                download_options['bulk_raw'] = bulk_zip
            
            # Enhanced bulk options
            enhanced_bulk = self._create_enhanced_bulk_options(artifacts, user_type)
            download_options['bulk_enhanced'] = enhanced_bulk
            
            # Combined report
            combined_report = self._create_combined_report(artifacts, user_type)
            if combined_report:
                download_options['combined_report'] = combined_report
            
            return download_options
            
        except Exception as e:
            logger.error(f"Error generating download options: {str(e)}")
            return {}
    
    def render_download_interface(self,
                                 artifacts: List[EnhancedArtifact],
                                 user_context: Optional[Dict[str, Any]] = None) -> None:
        """Render interactive download interface"""
        try:
            if not artifacts:
                st.info("다운로드할 아티팩트가 없습니다.")
                return
            
            st.markdown("## 📥 **다운로드 옵션**")
            
            # Generate options
            options = self.generate_download_options(artifacts, user_context)
            
            # Create tabs for different download types
            tab1, tab2, tab3 = st.tabs(["🔹 개별 다운로드", "📦 일괄 다운로드", "📋 통합 리포트"])
            
            with tab1:
                self._render_individual_downloads(options)
            
            with tab2:
                self._render_bulk_downloads(options)
            
            with tab3:
                self._render_combined_report_download(options)
                
        except Exception as e:
            logger.error(f"Error rendering download interface: {str(e)}")
            st.error("다운로드 인터페이스 오류")
    
    def _create_raw_download(self, artifact: EnhancedArtifact) -> Optional[Dict[str, Any]]:
        """Create raw download option for artifact"""
        try:
            raw_format = self.raw_formats.get(artifact.type, 'txt')
            
            # Prepare raw data
            if artifact.type == 'plotly_chart':
                data = json.dumps(artifact.data, indent=2).encode('utf-8')
                mime = 'application/json'
            elif artifact.type == 'table':
                if isinstance(artifact.data, pd.DataFrame):
                    data = artifact.data.to_csv(index=False).encode('utf-8')
                    mime = 'text/csv'
                else:
                    return None
            elif artifact.type == 'code':
                data = str(artifact.data).encode('utf-8')
                mime = 'text/plain'
            elif artifact.type == 'image':
                # Assume PIL Image or bytes
                if hasattr(artifact.data, 'save'):
                    buffer = io.BytesIO()
                    artifact.data.save(buffer, format='PNG')
                    data = buffer.getvalue()
                else:
                    data = artifact.data
                mime = 'image/png'
            elif artifact.type == 'markdown':
                data = str(artifact.data).encode('utf-8')
                mime = 'text/markdown'
            else:
                data = str(artifact.data).encode('utf-8')
                mime = 'text/plain'
            
            return {
                'label': f"Raw {artifact.title}",
                'data': data,
                'filename': f"{artifact.title}.{raw_format}",
                'mime': mime,
                'type': 'raw',
                'artifact_id': artifact.id
            }
            
        except Exception as e:
            logger.error(f"Error creating raw download: {str(e)}")
            return None
    
    def _create_enhanced_downloads(self,
                                  artifact: EnhancedArtifact,
                                  user_type: str) -> List[Dict[str, Any]]:
        """Create enhanced download options"""
        enhanced_options = []
        
        try:
            # User-specific enhanced formats
            target_formats = self.enhanced_formats.get(user_type, ['pdf', 'html'])
            
            for format_type in target_formats:
                enhanced_data = self._create_enhanced_format(artifact, format_type)
                if enhanced_data:
                    enhanced_options.append({
                        'label': f"{artifact.title} ({format_type.upper()})",
                        'data': enhanced_data['data'],
                        'filename': enhanced_data['filename'],
                        'mime': enhanced_data['mime'],
                        'type': 'enhanced',
                        'format': format_type,
                        'artifact_id': artifact.id
                    })
                    
        except Exception as e:
            logger.error(f"Error creating enhanced downloads: {str(e)}")
        
        return enhanced_options
    
    def _create_enhanced_format(self,
                               artifact: EnhancedArtifact,
                               format_type: str) -> Optional[Dict[str, Any]]:
        """Create specific enhanced format"""
        try:
            if format_type == 'pdf':
                return self._create_pdf_report(artifact)
            elif format_type == 'html':
                return self._create_html_report(artifact)
            elif format_type == 'xlsx':
                return self._create_excel_report(artifact)
            elif format_type == 'ipynb':
                return self._create_jupyter_notebook(artifact)
            elif format_type == 'pptx':
                return self._create_powerpoint(artifact)
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error creating {format_type} format: {str(e)}")
            return None
    
    def _create_pdf_report(self, artifact: EnhancedArtifact) -> Optional[Dict[str, Any]]:
        """Create PDF report (placeholder - requires reportlab)"""
        # This would require reportlab or similar PDF generation library
        # For now, return HTML that can be printed to PDF
        html_content = self._generate_report_html(artifact)
        
        return {
            'data': html_content.encode('utf-8'),
            'filename': f"{artifact.title}_report.html",
            'mime': 'text/html'
        }
    
    def _create_html_report(self, artifact: EnhancedArtifact) -> Dict[str, Any]:
        """Create HTML report"""
        html_content = self._generate_report_html(artifact)
        
        return {
            'data': html_content.encode('utf-8'),
            'filename': f"{artifact.title}_report.html",
            'mime': 'text/html'
        }
    
    def _create_excel_report(self, artifact: EnhancedArtifact) -> Optional[Dict[str, Any]]:
        """Create Excel report with formatting"""
        try:
            if not isinstance(artifact.data, pd.DataFrame):
                return None
            
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                # Main data sheet
                artifact.data.to_excel(writer, sheet_name='Data', index=False)
                
                # Metadata sheet
                metadata_df = pd.DataFrame([
                    ['제목', artifact.title],
                    ['설명', artifact.description or ''],
                    ['생성시간', artifact.created_at.strftime('%Y-%m-%d %H:%M:%S')],
                    ['타입', artifact.type],
                    ['행 수', len(artifact.data)],
                    ['열 수', len(artifact.data.columns)]
                ], columns=['항목', '값'])
                
                metadata_df.to_excel(writer, sheet_name='Metadata', index=False)
                
                # Apply formatting
                workbook = writer.book
                for sheet_name in workbook.sheetnames:
                    worksheet = workbook[sheet_name]
                    
                    # Header formatting
                    for cell in worksheet[1]:
                        cell.font = cell.font.copy(bold=True)
                        cell.fill = cell.fill.copy(fgColor="CCCCCC")
                    
                    # Auto-adjust column widths
                    for column in worksheet.columns:
                        max_length = 0
                        column_letter = column[0].column_letter
                        
                        for cell in column:
                            try:
                                if len(str(cell.value)) > max_length:
                                    max_length = len(str(cell.value))
                            except:
                                pass
                        
                        adjusted_width = min(max_length + 2, 50)
                        worksheet.column_dimensions[column_letter].width = adjusted_width
            
            return {
                'data': buffer.getvalue(),
                'filename': f"{artifact.title}_report.xlsx",
                'mime': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            }
            
        except Exception as e:
            logger.error(f"Error creating Excel report: {str(e)}")
            return None
    
    def _create_jupyter_notebook(self, artifact: EnhancedArtifact) -> Dict[str, Any]:
        """Create Jupyter notebook with artifact analysis"""
        try:
            notebook = {
                "cells": [
                    {
                        "cell_type": "markdown",
                        "metadata": {},
                        "source": [
                            f"# {artifact.title}\n\n",
                            f"**설명:** {artifact.description or 'Cherry AI 분석 결과'}\n\n",
                            f"**생성 시간:** {artifact.created_at.strftime('%Y-%m-%d %H:%M:%S')}\n\n",
                            f"**타입:** {artifact.type}\n\n"
                        ]
                    }
                ],
                "metadata": {
                    "kernelspec": {
                        "display_name": "Python 3",
                        "language": "python",
                        "name": "python3"
                    },
                    "language_info": {
                        "name": "python",
                        "version": "3.8.0"
                    }
                },
                "nbformat": 4,
                "nbformat_minor": 4
            }
            
            # Add code cell for data loading
            if isinstance(artifact.data, pd.DataFrame):
                code_cell = {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "import pandas as pd\n",
                        "import numpy as np\n",
                        "import matplotlib.pyplot as plt\n",
                        "import seaborn as sns\n\n",
                        "# Load data\n",
                        f"# {artifact.title}\n",
                        "# Data shape: " + str(artifact.data.shape) + "\n",
                        "df = pd.read_csv('data.csv')\n",
                        "df.head()"
                    ]
                }
                notebook["cells"].append(code_cell)
            
            notebook_json = json.dumps(notebook, indent=2)
            
            return {
                'data': notebook_json.encode('utf-8'),
                'filename': f"{artifact.title}_analysis.ipynb",
                'mime': 'application/json'
            }
            
        except Exception as e:
            logger.error(f"Error creating Jupyter notebook: {str(e)}")
            return None
    
    def _create_powerpoint(self, artifact: EnhancedArtifact) -> Optional[Dict[str, Any]]:
        """Create PowerPoint presentation (placeholder)"""
        # This would require python-pptx library
        # For now, return HTML that can be used for presentation
        html_content = self._generate_presentation_html(artifact)
        
        return {
            'data': html_content.encode('utf-8'),
            'filename': f"{artifact.title}_presentation.html",
            'mime': 'text/html'
        }
    
    def _create_bulk_zip(self, artifacts: List[EnhancedArtifact]) -> Optional[Dict[str, Any]]:
        """Create ZIP file with all raw artifacts"""
        try:
            buffer = io.BytesIO()
            
            with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                for artifact in artifacts:
                    raw_download = self._create_raw_download(artifact)
                    if raw_download:
                        zip_file.writestr(
                            raw_download['filename'],
                            raw_download['data']
                        )
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            return {
                'data': buffer.getvalue(),
                'filename': f"cherry_ai_artifacts_{timestamp}.zip",
                'mime': 'application/zip'
            }
            
        except Exception as e:
            logger.error(f"Error creating bulk ZIP: {str(e)}")
            return None
    
    def _create_enhanced_bulk_options(self,
                                     artifacts: List[EnhancedArtifact],
                                     user_type: str) -> List[Dict[str, Any]]:
        """Create enhanced bulk download options"""
        bulk_options = []
        
        try:
            # Combined Excel workbook
            excel_workbook = self._create_combined_excel(artifacts)
            if excel_workbook:
                bulk_options.append(excel_workbook)
            
            # Combined HTML report  
            html_report = self._create_combined_html_report(artifacts)
            if html_report:
                bulk_options.append(html_report)
                
        except Exception as e:
            logger.error(f"Error creating enhanced bulk options: {str(e)}")
        
        return bulk_options
    
    def _create_combined_excel(self, artifacts: List[EnhancedArtifact]) -> Optional[Dict[str, Any]]:
        """Create Excel workbook with all artifacts"""
        try:
            buffer = io.BytesIO()
            
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                for i, artifact in enumerate(artifacts):
                    if isinstance(artifact.data, pd.DataFrame):
                        sheet_name = f"Sheet_{i+1}_{artifact.title[:20]}"
                        artifact.data.to_excel(writer, sheet_name=sheet_name, index=False)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            return {
                'data': buffer.getvalue(),
                'filename': f"cherry_ai_combined_{timestamp}.xlsx",
                'mime': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            }
            
        except Exception as e:
            logger.error(f"Error creating combined Excel: {str(e)}")
            return None
    
    def _create_combined_html_report(self, artifacts: List[EnhancedArtifact]) -> Dict[str, Any]:
        """Create combined HTML report"""
        html_parts = [
            "<!DOCTYPE html>",
            "<html><head>",
            "<title>Cherry AI 종합 분석 리포트</title>",
            "<meta charset='utf-8'>",
            "<style>",
            "body { font-family: 'Segoe UI', sans-serif; margin: 20px; line-height: 1.6; }",
            ".header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 15px; text-align: center; }",
            ".artifact { margin: 30px 0; padding: 20px; border: 1px solid #ddd; border-radius: 10px; }",
            "table { border-collapse: collapse; width: 100%; margin: 15px 0; }",
            "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
            "th { background-color: #f2f2f2; }",
            "</style>",
            "</head><body>",
            f"<div class='header'><h1>🍒 Cherry AI 종합 분석 리포트</h1>",
            f"<p>생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p></div>"
        ]
        
        # Add each artifact
        for i, artifact in enumerate(artifacts, 1):
            html_parts.extend([
                f"<div class='artifact'>",
                f"<h2>{i}. {artifact.title}</h2>",
                f"<p><strong>설명:</strong> {artifact.description or '분석 결과'}</p>",
                f"<p><strong>타입:</strong> {artifact.type}</p>",
                f"<p><strong>생성 시간:</strong> {artifact.created_at.strftime('%Y-%m-%d %H:%M:%S')}</p>"
            ])
            
            # Add data representation
            if isinstance(artifact.data, pd.DataFrame):
                html_parts.append(artifact.data.head(10).to_html())
            elif isinstance(artifact.data, dict):
                html_parts.append(f"<pre>{json.dumps(artifact.data, indent=2, ensure_ascii=False)}</pre>")
            else:
                html_parts.append(f"<p>{str(artifact.data)[:500]}...</p>")
            
            html_parts.append("</div>")
        
        html_parts.extend([
            f"<div style='text-align: center; margin-top: 50px; color: #666;'>",
            f"<p>🍒 Cherry AI Platform | 총 {len(artifacts)}개의 분석 결과</p>",
            "</div>",
            "</body></html>"
        ])
        
        html_content = '\n'.join(html_parts)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        return {
            'data': html_content.encode('utf-8'),
            'filename': f"cherry_ai_report_{timestamp}.html",
            'mime': 'text/html'
        }
    
    def _create_combined_report(self, artifacts: List[EnhancedArtifact], user_type: str) -> Dict[str, Any]:
        """Create combined report based on user type"""
        if user_type == 'developer':
            return self._create_combined_html_report(artifacts)
        elif user_type in ['business_user', 'analyst']:
            return self._create_combined_excel(artifacts)
        else:
            return self._create_combined_html_report(artifacts)
    
    def _determine_user_type(self, user_context: Optional[Dict[str, Any]]) -> str:
        """Determine user type from context"""
        if not user_context:
            return 'analyst'  # Default
        
        # Simple heuristics
        role = user_context.get('role', '').lower()
        if 'developer' in role or 'engineer' in role:
            return 'developer'
        elif 'business' in role or 'manager' in role:
            return 'business_user'
        elif 'researcher' in role or 'scientist' in role:
            return 'researcher'
        else:
            return 'analyst'
    
    def _generate_report_html(self, artifact: EnhancedArtifact) -> str:
        """Generate HTML report for single artifact"""
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{artifact.title} - Cherry AI Analysis</title>
            <meta charset="utf-8">
            <style>
                body {{ font-family: 'Segoe UI', sans-serif; margin: 40px; line-height: 1.6; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; }}
                .content {{ margin-top: 20px; }}
                .metadata {{ background: #f8f9fa; padding: 15px; border-radius: 5px; margin: 10px 0; }}
                table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{artifact.title}</h1>
                <p>{artifact.description or 'Cherry AI 분석 결과'}</p>
            </div>
            
            <div class="content">
                <div class="metadata">
                    <strong>생성 시간:</strong> {artifact.created_at.strftime('%Y-%m-%d %H:%M:%S')}<br>
                    <strong>타입:</strong> {artifact.type}<br>
                    <strong>ID:</strong> {artifact.id}
                </div>
                
                {self._format_artifact_for_html(artifact)}
            </div>
            
            <div style="text-align: center; color: #666; margin-top: 50px;">
                Cherry AI Platform에서 생성됨 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            </div>
        </body>
        </html>
        """
        
        return html_template
    
    def _generate_presentation_html(self, artifact: EnhancedArtifact) -> str:
        """Generate presentation HTML for artifact"""
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{artifact.title} - Presentation</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 0; padding: 40px; }}
                .slide {{ min-height: 90vh; page-break-after: always; }}
                h1 {{ color: #667eea; }}
            </style>
        </head>
        <body>
            <div class="slide">
                <h1>{artifact.title}</h1>
                <p>{artifact.description or 'Cherry AI 분석 결과'}</p>
                {self._format_artifact_for_html(artifact)}
            </div>
        </body>
        </html>
        """
    
    def _format_artifact_for_html(self, artifact: EnhancedArtifact) -> str:
        """Format artifact data for HTML display"""
        if isinstance(artifact.data, pd.DataFrame):
            return artifact.data.to_html(classes='table')
        elif isinstance(artifact.data, dict):
            return f"<pre>{json.dumps(artifact.data, indent=2, ensure_ascii=False)}</pre>"
        else:
            return f"<p>{str(artifact.data)}</p>"
    
    def _render_individual_downloads(self, options: Dict[str, Any]) -> None:
        """Render individual download options"""
        st.markdown("### 개별 파일 다운로드")
        
        # Raw downloads
        if options.get('individual_raw'):
            st.markdown("#### 🔹 원본 데이터")
            for option in options['individual_raw']:
                st.download_button(
                    label=f"⬇️ {option['label']}",
                    data=option['data'],
                    file_name=option['filename'],
                    mime=option['mime'],
                    key=f"raw_{option['artifact_id']}"
                )
        
        # Enhanced downloads
        if options.get('individual_enhanced'):
            st.markdown("#### ✨ 향상된 형식")
            for option in options['individual_enhanced']:
                st.download_button(
                    label=f"⬇️ {option['label']}",
                    data=option['data'],
                    file_name=option['filename'],
                    mime=option['mime'],
                    key=f"enhanced_{option.get('format', 'unknown')}_{option['artifact_id']}"
                )
    
    def _render_bulk_downloads(self, options: Dict[str, Any]) -> None:
        """Render bulk download options"""
        st.markdown("### 일괄 다운로드")
        
        # Raw ZIP
        if options.get('bulk_raw'):
            option = options['bulk_raw']
            st.download_button(
                label="📦 모든 원본 파일 (ZIP)",
                data=option['data'],
                file_name=option['filename'],
                mime=option['mime'],
                key="bulk_raw_zip"
            )
        
        # Enhanced bulk options
        if options.get('bulk_enhanced'):
            for option in options['bulk_enhanced']:
                st.download_button(
                    label=f"📊 {option['filename']}",
                    data=option['data'],
                    file_name=option['filename'],
                    mime=option['mime'],
                    key=f"bulk_enhanced_{hash(option['filename'])}"
                )
    
    def _render_combined_report_download(self, options: Dict[str, Any]) -> None:
        """Render combined report download"""
        st.markdown("### 통합 리포트")
        
        if options.get('combined_report'):
            option = options['combined_report']
            st.download_button(
                label=f"📋 {option['filename']}",
                data=option['data'],
                file_name=option['filename'],
                mime=option['mime'],
                key="combined_report"
            )
        else:
            st.info("통합 리포트를 생성할 수 없습니다.")