"""
Interactive Plotly Renderer - Full Interactivity with Raw JSON Download

Comprehensive Plotly chart rendering with:
- Full interactivity (zoom, pan, hover)
- Export to PNG/SVG
- Raw JSON data always available
- Responsive design
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import json
import logging
from typing import Dict, Any, Optional, Union
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class InteractivePlotlyRenderer:
    """Interactive Plotly chart renderer with comprehensive features"""
    
    def __init__(self):
        """Initialize Plotly renderer"""
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
            }
        }
        
        # Theme configuration
        self.theme_colors = {
            'primary': '#667eea',
            'secondary': '#764ba2',
            'success': '#28a745',
            'warning': '#ffc107',
            'danger': '#dc3545',
            'info': '#17a2b8'
        }
    
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