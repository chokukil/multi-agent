"""
Interactive Plotly Renderer - Full Interactivity with Smart Download System

Enhanced Plotly rendering with smart download system:
- Full hover effects, zoom/pan functionality
- Responsive container sizing with aspect ratio preservation
- Streamlit theme integration with dark/light mode support
- Raw artifact download: Chart Data (JSON) - always available
- Context-aware enhanced formats: PNG/SVG for presentations, HTML for interactive sharing
- Chart customization controls (colors, layout, annotations)
- Click-to-enlarge functionality with modal display
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import json
import logging
from typing import Dict, Any, Optional, Union, List, Tuple
import pandas as pd
import numpy as np
import base64
from io import BytesIO
from datetime import datetime
import uuid

from ..interfaces import BaseRenderer
from ..models import EnhancedArtifact

logger = logging.getLogger(__name__)


class InteractivePlotlyRenderer(BaseRenderer):
    """Enhanced Plotly chart renderer with comprehensive features and smart downloads"""
    
    def __init__(self):
        """Initialize enhanced Plotly renderer"""
        super().__init__(supported_types=['plotly', 'plotly_chart', 'chart'])
        
        # Enhanced configuration
        self.default_config = {
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': ['sendDataToCloud'],
            'toImageButtonOptions': {
                'format': 'png',
                'filename': 'cherry_ai_chart',
                'height': 600,
                'width': 800,
                'scale': 2
            },
            'responsive': True,
            'showTips': True,
            'showAxisDragHandles': True,
            'showAxisRangeEntryBoxes': True
        }
        
        # Enhanced theme configuration
        self.theme_colors = {
            'primary': '#667eea',
            'secondary': '#764ba2',
            'success': '#28a745',
            'warning': '#ffc107',
            'danger': '#dc3545',
            'info': '#17a2b8',
            'light': '#f8f9fa',
            'dark': '#343a40'
        }
        
        # Chart type templates
        self.chart_templates = {
            'cherry_ai': {
                'layout': {
                    'colorway': [self.theme_colors['primary'], self.theme_colors['secondary'], 
                               self.theme_colors['success'], self.theme_colors['info']],
                    'font': {'family': 'Arial, sans-serif', 'size': 12},
                    'plot_bgcolor': 'rgba(0,0,0,0)',
                    'paper_bgcolor': 'rgba(0,0,0,0)',
                    'margin': {'l': 60, 'r': 60, 't': 80, 'b': 60}
                }
            }
        }
        
        logger.info("Enhanced Interactive Plotly Renderer initialized")
    
    def render_chart(self, 
                    chart_data: Union[go.Figure, Dict[str, Any]], 
                    title: Optional[str] = None,
                    height: int = 500,
                    use_container_width: bool = True) -> Dict[str, Any]:
        """
        Render interactive Plotly chart with download options
        
        Returns:
            Dict with 'raw_json' for download
        """
        try:
            # Convert to Plotly Figure if needed
            if isinstance(chart_data, dict):
                fig = self._dict_to_figure(chart_data)
            else:
                fig = chart_data
            
            # Apply Cherry AI theme
            fig = self._apply_theme(fig, title)
            
            # Update layout
            fig.update_layout(
                height=height,
                margin=dict(l=20, r=20, t=40, b=20),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            
            # Render in Streamlit
            st.plotly_chart(
                fig, 
                use_container_width=use_container_width,
                config=self.default_config
            )
            
            # Prepare raw JSON for download
            raw_json = fig.to_json()
            
            return {
                'raw_json': raw_json,
                'figure': fig
            }
            
        except Exception as e:
            logger.error(f"Error rendering Plotly chart: {str(e)}")
            st.error(f"차트 렌더링 오류: {str(e)}")
            return {'raw_json': '{}'}
    
    def _dict_to_figure(self, chart_dict: Dict[str, Any]) -> go.Figure:
        """Convert dictionary data to Plotly Figure"""
        
        chart_type = chart_dict.get('type', 'scatter')
        data = chart_dict.get('data', {})
        
        # Create appropriate chart based on type
        if chart_type == 'scatter':
            fig = self._create_scatter_plot(data)
        elif chart_type == 'bar':
            fig = self._create_bar_chart(data)
        elif chart_type == 'line':
            fig = self._create_line_chart(data)
        elif chart_type == 'heatmap':
            fig = self._create_heatmap(data)
        elif chart_type == 'box':
            fig = self._create_box_plot(data)
        elif chart_type == 'histogram':
            fig = self._create_histogram(data)
        elif chart_type == 'pie':
            fig = self._create_pie_chart(data)
        elif chart_type == '3d_scatter':
            fig = self._create_3d_scatter(data)
        else:
            # Default to scatter plot
            fig = self._create_scatter_plot(data)
        
        return fig
    
    def _apply_theme(self, fig: go.Figure, title: Optional[str] = None) -> go.Figure:
        """Apply Cherry AI theme to figure"""
        
        # Update layout with theme
        fig.update_layout(
            title={
                'text': title or fig.layout.title.text,
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'color': self.theme_colors['primary']}
            },
            font={'family': 'Segoe UI, sans-serif', 'size': 12},
            colorway=[
                self.theme_colors['primary'],
                self.theme_colors['secondary'],
                self.theme_colors['success'],
                self.theme_colors['warning'],
                self.theme_colors['danger'],
                self.theme_colors['info']
            ],
            hovermode='x unified',
            hoverlabel=dict(
                bgcolor="white",
                font_size=12,
                font_family="Segoe UI"
            )
        )
        
        # Update axes
        fig.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.2)',
            showline=True,
            linewidth=1,
            linecolor='rgba(128,128,128,0.3)'
        )
        
        fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.2)',
            showline=True,
            linewidth=1,
            linecolor='rgba(128,128,128,0.3)'
        )
        
        return fig
    
    def _create_scatter_plot(self, data: Dict[str, Any]) -> go.Figure:
        """Create scatter plot"""
        x_data = data.get('x', [])
        y_data = data.get('y', [])
        
        fig = go.Figure()
        
        # Support multiple series
        if isinstance(x_data, dict):
            for series_name, x_values in x_data.items():
                y_values = y_data.get(series_name, [])
                fig.add_trace(go.Scatter(
                    x=x_values,
                    y=y_values,
                    mode='markers',
                    name=series_name,
                    marker=dict(size=8)
                ))
        else:
            fig.add_trace(go.Scatter(
                x=x_data,
                y=y_data,
                mode='markers',
                marker=dict(
                    size=8,
                    color=self.theme_colors['primary']
                )
            ))
        
        fig.update_layout(
            xaxis_title=data.get('x_label', 'X'),
            yaxis_title=data.get('y_label', 'Y')
        )
        
        return fig
    
    def _create_bar_chart(self, data: Dict[str, Any]) -> go.Figure:
        """Create bar chart"""
        x_data = data.get('x', [])
        y_data = data.get('y', [])
        
        fig = go.Figure()
        
        if isinstance(y_data, dict):
            # Multiple series
            for series_name, values in y_data.items():
                fig.add_trace(go.Bar(
                    x=x_data,
                    y=values,
                    name=series_name
                ))
        else:
            # Single series
            fig.add_trace(go.Bar(
                x=x_data,
                y=y_data,
                marker_color=self.theme_colors['primary']
            ))
        
        fig.update_layout(
            xaxis_title=data.get('x_label', 'Categories'),
            yaxis_title=data.get('y_label', 'Values'),
            barmode=data.get('barmode', 'group')
        )
        
        return fig
    
    def _create_line_chart(self, data: Dict[str, Any]) -> go.Figure:
        """Create line chart"""
        x_data = data.get('x', [])
        y_data = data.get('y', [])
        
        fig = go.Figure()
        
        if isinstance(y_data, dict):
            # Multiple lines
            for series_name, values in y_data.items():
                fig.add_trace(go.Scatter(
                    x=x_data,
                    y=values,
                    mode='lines+markers',
                    name=series_name,
                    line=dict(width=2),
                    marker=dict(size=6)
                ))
        else:
            # Single line
            fig.add_trace(go.Scatter(
                x=x_data,
                y=y_data,
                mode='lines+markers',
                line=dict(color=self.theme_colors['primary'], width=2),
                marker=dict(size=6)
            ))
        
        fig.update_layout(
            xaxis_title=data.get('x_label', 'X'),
            yaxis_title=data.get('y_label', 'Y')
        )
        
        return fig
    
    def _create_heatmap(self, data: Dict[str, Any]) -> go.Figure:
        """Create heatmap"""
        z_data = data.get('z', [[]])
        x_labels = data.get('x_labels', [])
        y_labels = data.get('y_labels', [])
        
        fig = go.Figure(data=go.Heatmap(
            z=z_data,
            x=x_labels,
            y=y_labels,
            colorscale='Viridis',
            text=z_data,
            texttemplate='%{text:.2f}',
            textfont={"size": 10}
        ))
        
        fig.update_layout(
            xaxis_title=data.get('x_label', 'X'),
            yaxis_title=data.get('y_label', 'Y')
        )
        
        return fig
    
    def _create_box_plot(self, data: Dict[str, Any]) -> go.Figure:
        """Create box plot"""
        fig = go.Figure()
        
        if isinstance(data.get('values'), dict):
            # Multiple box plots
            for name, values in data['values'].items():
                fig.add_trace(go.Box(
                    y=values,
                    name=name,
                    boxpoints='outliers'
                ))
        else:
            # Single box plot
            fig.add_trace(go.Box(
                y=data.get('values', []),
                name=data.get('name', 'Values'),
                boxpoints='outliers',
                marker_color=self.theme_colors['primary']
            ))
        
        fig.update_layout(
            yaxis_title=data.get('y_label', 'Values')
        )
        
        return fig
    
    def _create_histogram(self, data: Dict[str, Any]) -> go.Figure:
        """Create histogram"""
        values = data.get('values', [])
        bins = data.get('bins', 30)
        
        fig = go.Figure(data=go.Histogram(
            x=values,
            nbinsx=bins,
            marker_color=self.theme_colors['primary']
        ))
        
        fig.update_layout(
            xaxis_title=data.get('x_label', 'Values'),
            yaxis_title=data.get('y_label', 'Frequency'),
            bargap=0.1
        )
        
        return fig
    
    def _create_pie_chart(self, data: Dict[str, Any]) -> go.Figure:
        """Create pie chart"""
        labels = data.get('labels', [])
        values = data.get('values', [])
        
        fig = go.Figure(data=go.Pie(
            labels=labels,
            values=values,
            hole=0.3,  # Donut chart
            textinfo='label+percent',
            textposition='outside'
        ))
        
        fig.update_traces(
            hoverinfo='label+percent+value',
            textfont_size=12
        )
        
        return fig
    
    def _create_3d_scatter(self, data: Dict[str, Any]) -> go.Figure:
        """Create 3D scatter plot"""
        x = data.get('x', [])
        y = data.get('y', [])
        z = data.get('z', [])
        
        fig = go.Figure(data=go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='markers',
            marker=dict(
                size=5,
                color=z,
                colorscale='Viridis',
                opacity=0.8,
                showscale=True
            )
        ))
        
        fig.update_layout(
            scene=dict(
                xaxis_title=data.get('x_label', 'X'),
                yaxis_title=data.get('y_label', 'Y'),
                zaxis_title=data.get('z_label', 'Z')
            )
        )
        
        return fig
    
    def render_dataframe_chart(self, 
                              df: pd.DataFrame,
                              chart_type: str = 'scatter',
                              x_col: Optional[str] = None,
                              y_col: Optional[str] = None,
                              color_col: Optional[str] = None,
                              size_col: Optional[str] = None,
                              title: Optional[str] = None) -> Dict[str, Any]:
        """Render chart from DataFrame"""
        try:
            # Select appropriate Plotly Express function
            if chart_type == 'scatter':
                fig = px.scatter(df, x=x_col, y=y_col, color=color_col, size=size_col, title=title)
            elif chart_type == 'line':
                fig = px.line(df, x=x_col, y=y_col, color=color_col, title=title)
            elif chart_type == 'bar':
                fig = px.bar(df, x=x_col, y=y_col, color=color_col, title=title)
            elif chart_type == 'histogram':
                fig = px.histogram(df, x=x_col, color=color_col, title=title)
            elif chart_type == 'box':
                fig = px.box(df, x=x_col, y=y_col, color=color_col, title=title)
            elif chart_type == 'violin':
                fig = px.violin(df, x=x_col, y=y_col, color=color_col, title=title)
            elif chart_type == 'heatmap':
                # For heatmap, assume pivot table format
                fig = px.imshow(df.values, x=df.columns, y=df.index, title=title)
            else:
                fig = px.scatter(df, x=x_col, y=y_col, color=color_col, title=title)
            
            # Apply theme
            fig = self._apply_theme(fig, title)
            
            # Render
            return self.render_chart(fig)
            
        except Exception as e:
            logger.error(f"Error rendering DataFrame chart: {str(e)}")
            st.error(f"DataFrame 차트 렌더링 오류: {str(e)}")
            return {'raw_json': '{}'}    

    def render_artifact(self, artifact: EnhancedArtifact) -> None:
        """
        Render enhanced Plotly chart with comprehensive features:
        - Full interactivity with hover, zoom, pan
        - Responsive design with aspect ratio preservation
        - Theme integration and customization controls
        - Smart download system with raw and enhanced formats
        """
        try:
            # Extract chart data
            chart_data = self._extract_chart_data(artifact)
            
            # Create enhanced figure
            fig = self._create_enhanced_figure(chart_data, artifact)
            
            # Render chart with controls
            self._render_chart_with_controls(fig, artifact)
            
            # Render download options
            self._render_download_options(fig, artifact)
            
        except Exception as e:
            logger.error(f"Error rendering Plotly artifact: {str(e)}")
            st.error(f"❌ Chart rendering failed: {str(e)}")
            
            # Fallback to raw data display
            with st.expander("📄 Raw Chart Data", expanded=False):
                st.json(artifact.data)
    
    def _extract_chart_data(self, artifact: EnhancedArtifact) -> Dict[str, Any]:
        """Extract and validate chart data from artifact"""
        try:
            if isinstance(artifact.data, dict):
                return artifact.data
            elif isinstance(artifact.data, str):
                return json.loads(artifact.data)
            else:
                raise ValueError("Invalid chart data format")
                
        except Exception as e:
            logger.error(f"Error extracting chart data: {str(e)}")
            raise ValueError(f"Failed to parse chart data: {str(e)}")
    
    def _create_enhanced_figure(self, chart_data: Dict[str, Any], artifact: EnhancedArtifact) -> go.Figure:
        """Create enhanced Plotly figure with theme and customizations"""
        try:
            # Create figure from data
            if 'data' in chart_data and 'layout' in chart_data:
                # Standard Plotly JSON format
                fig = go.Figure(data=chart_data['data'], layout=chart_data['layout'])
            else:
                # Try to create from raw data
                fig = self._create_figure_from_raw_data(chart_data)
            
            # Apply Cherry AI theme
            self._apply_cherry_ai_theme(fig)
            
            # Add responsive layout
            self._make_responsive(fig)
            
            # Add interactive features
            self._enhance_interactivity(fig)
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating enhanced figure: {str(e)}")
            # Create fallback figure
            return self._create_fallback_figure(chart_data)
    
    def _create_figure_from_raw_data(self, data: Dict[str, Any]) -> go.Figure:
        """Create figure from raw data when standard format is not available"""
        
        # Try to detect data structure and create appropriate chart
        if 'x' in data and 'y' in data:
            # Simple x-y data
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=data['x'],
                y=data['y'],
                mode='lines+markers',
                name=data.get('name', 'Data')
            ))
        elif 'values' in data and 'labels' in data:
            # Pie chart data
            fig = go.Figure()
            fig.add_trace(go.Pie(
                values=data['values'],
                labels=data['labels'],
                name=data.get('name', 'Data')
            ))
        else:
            # Generic bar chart
            keys = list(data.keys())
            if len(keys) >= 2:
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=data[keys[0]],
                    y=data[keys[1]],
                    name=data.get('name', 'Data')
                ))
            else:
                raise ValueError("Unable to determine chart type from data")
        
        return fig
    
    def _apply_cherry_ai_theme(self, fig: go.Figure):
        """Apply Cherry AI theme to figure"""
        
        # Update layout with theme
        fig.update_layout(
            template='plotly_white',
            colorway=[
                self.theme_colors['primary'],
                self.theme_colors['secondary'],
                self.theme_colors['success'],
                self.theme_colors['info'],
                self.theme_colors['warning'],
                self.theme_colors['danger']
            ],
            font=dict(
                family="Arial, sans-serif",
                size=12,
                color="#2c3e50"
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=60, r=60, t=80, b=60),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Update axes
        fig.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(0,0,0,0.1)',
            showline=True,
            linewidth=1,
            linecolor='rgba(0,0,0,0.3)'
        )
        
        fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(0,0,0,0.1)',
            showline=True,
            linewidth=1,
            linecolor='rgba(0,0,0,0.3)'
        )
    
    def _make_responsive(self, fig: go.Figure):
        """Make figure responsive with proper aspect ratio"""
        
        fig.update_layout(
            autosize=True,
            height=500,  # Default height
            margin=dict(l=60, r=60, t=80, b=60)
        )
    
    def _enhance_interactivity(self, fig: go.Figure):
        """Enhance figure interactivity"""
        
        # Add hover template improvements
        for trace in fig.data:
            if hasattr(trace, 'hovertemplate'):
                # Enhance hover information
                if trace.hovertemplate is None:
                    if trace.type == 'scatter':
                        trace.hovertemplate = '<b>%{fullData.name}</b><br>X: %{x}<br>Y: %{y}<extra></extra>'
                    elif trace.type == 'bar':
                        trace.hovertemplate = '<b>%{fullData.name}</b><br>Category: %{x}<br>Value: %{y}<extra></extra>'
        
        # Configure zoom and pan
        fig.update_layout(
            dragmode='zoom',
            selectdirection='diagonal'
        )
    
    def _render_chart_with_controls(self, fig: go.Figure, artifact: EnhancedArtifact):
        """Render chart with interactive controls"""
        
        # Chart title and description
        st.markdown(f"### 📊 {artifact.title}")
        if artifact.description:
            st.markdown(f"*{artifact.description}*")
        
        # Chart customization controls
        with st.expander("🎨 Chart Customization", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Height control
                height = st.slider(
                    "Chart Height",
                    min_value=300,
                    max_value=800,
                    value=500,
                    step=50,
                    key=f"height_{artifact.id}"
                )
                fig.update_layout(height=height)
            
            with col2:
                # Theme selection
                theme = st.selectbox(
                    "Theme",
                    ["plotly_white", "plotly_dark", "ggplot2", "seaborn"],
                    key=f"theme_{artifact.id}"
                )
                fig.update_layout(template=theme)
            
            with col3:
                # Show/hide legend
                show_legend = st.checkbox(
                    "Show Legend",
                    value=True,
                    key=f"legend_{artifact.id}"
                )
                fig.update_layout(showlegend=show_legend)
        
        # Render the chart
        st.plotly_chart(
            fig,
            use_container_width=True,
            config=self.default_config,
            key=f"chart_{artifact.id}"
        )
        
        # Chart statistics
        self._render_chart_statistics(fig, artifact)
    
    def _render_chart_statistics(self, fig: go.Figure, artifact: EnhancedArtifact):
        """Render chart statistics and metadata"""
        
        with st.expander("📈 Chart Information", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Chart Details:**")
                st.write(f"• Type: {fig.data[0].type if fig.data else 'Unknown'}")
                st.write(f"• Traces: {len(fig.data)}")
                st.write(f"• Created: {artifact.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
                st.write(f"• Size: {artifact.file_size_mb:.2f} MB")
            
            with col2:
                st.markdown("**Data Points:**")
                total_points = 0
                for trace in fig.data:
                    if hasattr(trace, 'x') and trace.x is not None:
                        total_points += len(trace.x)
                    elif hasattr(trace, 'values') and trace.values is not None:
                        total_points += len(trace.values)
                
                st.write(f"• Total Points: {total_points:,}")
                st.write(f"• Interactive: ✅")
                st.write(f"• Responsive: ✅")
                st.write(f"• Downloadable: ✅")
    
    def _render_download_options(self, fig: go.Figure, artifact: EnhancedArtifact):
        """Render smart download system with two-tier approach"""
        
        st.markdown("### 💾 Download Options")
        
        # Two-tier download system
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🔒 Raw Artifacts (Always Available)")
            
            # Raw JSON data (always available)
            raw_json = json.dumps(fig.to_dict(), indent=2)
            st.download_button(
                label="📊 Chart Data (JSON)",
                data=raw_json,
                file_name=f"{artifact.title.replace(' ', '_')}_data.json",
                mime="application/json",
                help="Raw chart data in JSON format"
            )
            
            # Raw figure object
            fig_json = json.dumps(fig.to_json(), indent=2)
            st.download_button(
                label="📈 Figure Object (JSON)",
                data=fig_json,
                file_name=f"{artifact.title.replace(' ', '_')}_figure.json",
                mime="application/json",
                help="Complete Plotly figure object"
            )
        
        with col2:
            st.markdown("#### 🎨 Enhanced Formats (Context-Based)")
            
            # PNG export
            png_data = self._export_to_png(fig)
            if png_data:
                st.download_button(
                    label="🖼️ High-Res PNG",
                    data=png_data,
                    file_name=f"{artifact.title.replace(' ', '_')}.png",
                    mime="image/png",
                    help="High-resolution PNG for presentations"
                )
            
            # SVG export
            svg_data = self._export_to_svg(fig)
            if svg_data:
                st.download_button(
                    label="🎨 Vector SVG",
                    data=svg_data,
                    file_name=f"{artifact.title.replace(' ', '_')}.svg",
                    mime="image/svg+xml",
                    help="Scalable vector graphics for web"
                )
            
            # Interactive HTML
            html_data = self._export_to_html(fig, artifact)
            if html_data:
                st.download_button(
                    label="🌐 Interactive HTML",
                    data=html_data,
                    file_name=f"{artifact.title.replace(' ', '_')}.html",
                    mime="text/html",
                    help="Standalone interactive HTML file"
                )
    
    def _export_to_png(self, fig: go.Figure, width: int = 1200, height: int = 800) -> Optional[bytes]:
        """Export figure to high-resolution PNG"""
        try:
            img_bytes = pio.to_image(
                fig,
                format="png",
                width=width,
                height=height,
                scale=2  # High DPI
            )
            return img_bytes
            
        except Exception as e:
            logger.error(f"PNG export failed: {str(e)}")
            st.error("PNG export failed. Please try again.")
            return None
    
    def _export_to_svg(self, fig: go.Figure) -> Optional[str]:
        """Export figure to SVG format"""
        try:
            svg_string = pio.to_image(fig, format="svg").decode('utf-8')
            return svg_string
            
        except Exception as e:
            logger.error(f"SVG export failed: {str(e)}")
            st.error("SVG export failed. Please try again.")
            return None
    
    def _export_to_html(self, fig: go.Figure, artifact: EnhancedArtifact) -> Optional[str]:
        """Export figure to standalone HTML"""
        try:
            html_string = pio.to_html(
                fig,
                include_plotlyjs=True,
                div_id=f"chart_{artifact.id}",
                config=self.default_config
            )
            
            # Add Cherry AI branding
            branded_html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{artifact.title} - Cherry AI</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f8f9fa;
        }}
        .header {{
            text-align: center;
            margin-bottom: 20px;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
        }}
        .chart-container {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🍒 Cherry AI</h1>
        <h2>{artifact.title}</h2>
        <p>{artifact.description}</p>
        <small>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</small>
    </div>
    <div class="chart-container">
        {html_string}
    </div>
</body>
</html>
"""
            return branded_html
            
        except Exception as e:
            logger.error(f"HTML export failed: {str(e)}")
            st.error("HTML export failed. Please try again.")
            return None
    
    def _create_fallback_figure(self, data: Dict[str, Any]) -> go.Figure:
        """Create fallback figure when main creation fails"""
        
        fig = go.Figure()
        
        # Add a simple text annotation
        fig.add_annotation(
            text="Chart data could not be rendered<br>Please check the raw data below",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="red")
        )
        
        fig.update_layout(
            title="Chart Rendering Error",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            height=300
        )
        
        return fig
    
    def get_download_options(self, artifact: EnhancedArtifact) -> List[Dict[str, Any]]:
        """Get available download options for the artifact"""
        
        options = [
            {
                'name': 'Chart Data (JSON)',
                'description': 'Raw chart data in JSON format',
                'format': 'json',
                'type': 'raw',
                'icon': '📊',
                'always_available': True
            },
            {
                'name': 'Figure Object (JSON)', 
                'description': 'Complete Plotly figure object',
                'format': 'json',
                'type': 'raw',
                'icon': '📈',
                'always_available': True
            },
            {
                'name': 'High-Res PNG',
                'description': 'High-resolution PNG for presentations',
                'format': 'png',
                'type': 'enhanced',
                'icon': '🖼️',
                'context': 'presentation'
            },
            {
                'name': 'Vector SVG',
                'description': 'Scalable vector graphics for web',
                'format': 'svg', 
                'type': 'enhanced',
                'icon': '🎨',
                'context': 'web'
            },
            {
                'name': 'Interactive HTML',
                'description': 'Standalone interactive HTML file',
                'format': 'html',
                'type': 'enhanced', 
                'icon': '🌐',
                'context': 'sharing'
            }
        ]
        
        return options
    
    def supports_artifact_type(self, artifact_type: str) -> bool:
        """Check if renderer supports the given artifact type"""
        supported_types = ['plotly', 'plotly_chart', 'chart', 'visualization']
        return artifact_type.lower() in supported_types
    
    def get_renderer_info(self) -> Dict[str, Any]:
        """Get renderer information"""
        return {
            'name': 'Interactive Plotly Renderer',
            'version': '1.0.0',
            'supported_types': self.get_supported_types(),
            'features': [
                'Full interactivity (zoom, pan, hover)',
                'Responsive design',
                'Theme customization',
                'Smart download system',
                'High-resolution exports',
                'Vector graphics support',
                'Standalone HTML export'
            ],
            'raw_formats': ['json'],
            'enhanced_formats': ['png', 'svg', 'html']
        }