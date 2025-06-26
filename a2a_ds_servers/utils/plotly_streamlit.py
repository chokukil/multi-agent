# A2A Data Science Servers - Streamlit Plotly Utilities
# Advanced Plotly integration optimized for Streamlit dashboards

import json
import plotly.io as pio
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, Any, Optional, List, Union
import streamlit as st
from datetime import datetime

def plotly_from_dict(plotly_graph_dict: dict):
    """
    Convert a Plotly graph dictionary to a Plotly graph object.
    Enhanced version with error handling and Streamlit optimization.
    
    Parameters:
    -----------
    plotly_graph_dict: dict
        A Plotly graph dictionary.
        
    Returns:
    --------
    plotly_graph: plotly.graph_objs.Figure
        A Plotly graph object optimized for Streamlit.
    """
    
    if plotly_graph_dict is None:
        return None
    
    try:
        fig = pio.from_json(json.dumps(plotly_graph_dict))
        # Optimize for Streamlit
        return optimize_for_streamlit(fig)
    except Exception as e:
        # Fallback: create a simple error chart
        error_fig = go.Figure()
        error_fig.add_annotation(
            text=f"Error loading chart: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return error_fig

def optimize_for_streamlit(fig: go.Figure) -> go.Figure:
    """
    Optimize a Plotly figure for Streamlit display.
    
    Parameters:
    -----------
    fig: plotly.graph_objects.Figure
        The figure to optimize.
        
    Returns:
    --------
    optimized_fig: plotly.graph_objects.Figure
        Streamlit-optimized figure.
    """
    
    # Update layout for better Streamlit integration
    fig.update_layout(
        # Responsive design
        autosize=True,
        
        # Dark theme compatibility
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        
        # Better margins for Streamlit
        margin=dict(l=20, r=20, t=40, b=20),
        
        # Modern font
        font=dict(
            family="system-ui, -apple-system, sans-serif",
            size=12,
            color="inherit"
        ),
        
        # Interactive features
        hovermode='closest',
        
        # Better legend positioning
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Update traces for better interactivity
    fig.update_traces(
        hovertemplate='<b>%{fullData.name}</b><br>%{x}: %{y}<extra></extra>'
    )
    
    return fig

def streamlit_plotly_chart(
    fig: go.Figure, 
    title: Optional[str] = None,
    use_container_width: bool = True,
    height: Optional[int] = None,
    config: Optional[Dict[str, Any]] = None
) -> None:
    """
    Display a Plotly chart in Streamlit with enhanced features.
    
    Parameters:
    -----------
    fig: plotly.graph_objects.Figure
        The figure to display.
    title: str, optional
        Title for the chart section.
    use_container_width: bool
        Whether to use container width.
    height: int, optional
        Height of the chart.
    config: dict, optional
        Plotly config options.
    """
    
    if title:
        st.subheader(title)
    
    # Default config for better UX
    default_config = {
        'displayModeBar': True,
        'displaylogo': False,
        'modeBarButtonsToRemove': ['pan2d', 'lasso2d'],
        'toImageButtonOptions': {
            'format': 'png',
            'filename': f'chart_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            'height': 600,
            'width': 1000,
            'scale': 2
        }
    }
    
    if config:
        default_config.update(config)
    
    # Display the chart
    st.plotly_chart(
        fig, 
        use_container_width=use_container_width,
        height=height,
        config=default_config
    )

def create_interactive_dashboard(
    data: pd.DataFrame,
    chart_types: List[str] = None,
    filters: Dict[str, Any] = None
) -> Dict[str, go.Figure]:
    """
    Create an interactive dashboard with multiple chart types.
    
    Parameters:
    -----------
    data: pd.DataFrame
        The data to visualize.
    chart_types: List[str]
        Types of charts to create.
    filters: Dict[str, Any]
        Filter configurations.
        
    Returns:
    --------
    charts: Dict[str, go.Figure]
        Dictionary of created charts.
    """
    
    if chart_types is None:
        chart_types = ['scatter', 'bar', 'line', 'histogram']
    
    charts = {}
    
    # Get numeric columns
    numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
    
    if len(numeric_cols) >= 2:
        # Scatter plot
        if 'scatter' in chart_types:
            charts['scatter'] = px.scatter(
                data, 
                x=numeric_cols[0], 
                y=numeric_cols[1],
                color=categorical_cols[0] if categorical_cols else None,
                title="Scatter Plot Analysis"
            )
        
        # Bar chart
        if 'bar' in chart_types and categorical_cols:
            charts['bar'] = px.bar(
                data.groupby(categorical_cols[0])[numeric_cols[0]].mean().reset_index(),
                x=categorical_cols[0],
                y=numeric_cols[0],
                title=f"Average {numeric_cols[0]} by {categorical_cols[0]}"
            )
        
        # Line chart
        if 'line' in chart_types:
            if 'date' in data.columns or 'time' in data.columns:
                date_col = 'date' if 'date' in data.columns else 'time'
                charts['line'] = px.line(
                    data, 
                    x=date_col, 
                    y=numeric_cols[0],
                    title="Time Series Analysis"
                )
            else:
                charts['line'] = px.line(
                    data.head(50), 
                    x=data.index[:50], 
                    y=numeric_cols[0],
                    title="Sequential Data Analysis"
                )
        
        # Histogram
        if 'histogram' in chart_types:
            charts['histogram'] = px.histogram(
                data, 
                x=numeric_cols[0],
                title=f"Distribution of {numeric_cols[0]}"
            )
    
    # Optimize all charts for Streamlit
    for key, fig in charts.items():
        charts[key] = optimize_for_streamlit(fig)
    
    return charts

def create_correlation_heatmap(data: pd.DataFrame) -> go.Figure:
    """
    Create an interactive correlation heatmap.
    
    Parameters:
    -----------
    data: pd.DataFrame
        The data to analyze.
        
    Returns:
    --------
    fig: plotly.graph_objects.Figure
        Correlation heatmap figure.
    """
    
    # Calculate correlation matrix
    numeric_data = data.select_dtypes(include=['number'])
    correlation_matrix = numeric_data.corr()
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=correlation_matrix.round(2).values,
        texttemplate="%{text}",
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title="Correlation Matrix",
        xaxis_title="Features",
        yaxis_title="Features"
    )
    
    return optimize_for_streamlit(fig)

def create_distribution_plots(data: pd.DataFrame) -> go.Figure:
    """
    Create distribution plots for numeric columns.
    
    Parameters:
    -----------
    data: pd.DataFrame
        The data to analyze.
        
    Returns:
    --------
    fig: plotly.graph_objects.Figure
        Distribution plots figure.
    """
    
    numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
    
    if not numeric_cols:
        return go.Figure().add_annotation(
            text="No numeric columns found",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
    
    # Create subplots
    num_cols = min(len(numeric_cols), 4)
    rows = (len(numeric_cols) + num_cols - 1) // num_cols
    
    fig = make_subplots(
        rows=rows, 
        cols=num_cols,
        subplot_titles=numeric_cols[:rows*num_cols]
    )
    
    for i, col in enumerate(numeric_cols[:rows*num_cols]):
        row = i // num_cols + 1
        col_pos = i % num_cols + 1
        
        fig.add_trace(
            go.Histogram(x=data[col], name=col),
            row=row, col=col_pos
        )
    
    fig.update_layout(
        title="Distribution Analysis",
        showlegend=False
    )
    
    return optimize_for_streamlit(fig)

def export_chart_data(fig: go.Figure, filename: str = None) -> Dict[str, Any]:
    """
    Export chart data for artifacts.
    
    Parameters:
    -----------
    fig: plotly.graph_objects.Figure
        The figure to export.
    filename: str, optional
        Filename for export.
        
    Returns:
    --------
    artifact: Dict[str, Any]
        Chart artifact data.
    """
    
    if filename is None:
        filename = f"chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Convert to JSON
    chart_json = fig.to_json()
    
    return {
        "type": "plotly_chart",
        "filename": f"{filename}.json",
        "data": json.loads(chart_json),
        "timestamp": datetime.now().isoformat(),
        "metadata": {
            "chart_type": "plotly",
            "traces": len(fig.data),
            "layout_keys": list(fig.layout.keys())
        }
    } 