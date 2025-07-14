"""
Automatic visualization generation for pandas_agent

This module provides intelligent chart generation based on data characteristics
and user queries, using matplotlib and seaborn.
"""

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io
import base64
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime
import warnings

# Configure matplotlib for headless environments
matplotlib.use('Agg')
plt.style.use('default')
sns.set_theme()

import sys
import os
from pathlib import Path

# Add pandas_agent directory to Python path
pandas_agent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(pandas_agent_dir))

from helpers.logger import get_logger


class AutoVisualizationEngine:
    """
    Automatic visualization generation engine
    
    Intelligently creates appropriate visualizations based on:
    - Data types and characteristics
    - User query intent
    - Statistical properties
    - Best practices for data visualization
    """
    
    def __init__(self, 
                 style: str = 'whitegrid',
                 color_palette: str = 'husl',
                 figure_size: Tuple[int, int] = (10, 6),
                 dpi: int = 100):
        """
        Initialize visualization engine
        
        Args:
            style: Seaborn style theme
            color_palette: Default color palette
            figure_size: Default figure size
            dpi: Figure DPI for quality
        """
        self.style = style
        self.color_palette = color_palette
        self.figure_size = figure_size
        self.dpi = dpi
        self.logger = get_logger()
        
        # Configure default settings
        self._configure_plotting_defaults()
        
        # Visualization templates
        self.chart_templates = {
            'histogram': self._create_histogram,
            'scatter': self._create_scatter_plot,
            'line': self._create_line_plot,
            'bar': self._create_bar_chart,
            'box': self._create_box_plot,
            'violin': self._create_violin_plot,
            'heatmap': self._create_heatmap,
            'correlation': self._create_correlation_matrix,
            'distribution': self._create_distribution_plot,
            'count': self._create_count_plot,
            'regression': self._create_regression_plot,
            'pair': self._create_pair_plot
        }
        
        self.logger.info("AutoVisualizationEngine initialized")
    
    def _configure_plotting_defaults(self):
        """Configure default plotting settings"""
        try:
            sns.set_style(self.style)
            sns.set_palette(self.color_palette)
            plt.rcParams['figure.figsize'] = self.figure_size
            plt.rcParams['figure.dpi'] = self.dpi
            plt.rcParams['font.size'] = 10
            plt.rcParams['axes.titlesize'] = 12
            plt.rcParams['axes.labelsize'] = 10
            plt.rcParams['xtick.labelsize'] = 9
            plt.rcParams['ytick.labelsize'] = 9
            plt.rcParams['legend.fontsize'] = 9
            
            # Suppress warnings
            warnings.filterwarnings('ignore', category=UserWarning)
            
        except Exception as e:
            self.logger.warning(f"Could not configure plotting defaults: {e}")
    
    def generate_auto_visualizations(self, 
                                   df: pd.DataFrame,
                                   query: str = "",
                                   intent: str = "exploration",
                                   max_charts: int = 3) -> Dict[str, Any]:
        """
        Generate appropriate visualizations automatically
        
        Args:
            df: DataFrame to visualize
            query: User query for context
            intent: Analysis intent (exploration, correlation, etc.)
            max_charts: Maximum number of charts to generate
            
        Returns:
            Dictionary with generated visualizations
        """
        try:
            self.logger.info(f"Generating auto visualizations for intent: {intent}")
            
            # Analyze data characteristics
            data_analysis = self._analyze_data_for_visualization(df)
            
            # Determine appropriate chart types
            recommended_charts = self._recommend_chart_types(
                data_analysis, query, intent
            )
            
            # Generate visualizations
            generated_charts = []
            for i, chart_info in enumerate(recommended_charts[:max_charts]):
                try:
                    chart_result = self._generate_single_chart(
                        df, chart_info, data_analysis
                    )
                    if chart_result:
                        generated_charts.append(chart_result)
                        
                except Exception as e:
                    self.logger.warning(f"Failed to generate chart {chart_info['type']}: {e}")
                    continue
            
            return {
                "charts": generated_charts,
                "data_analysis": data_analysis,
                "recommendations": self._generate_visualization_recommendations(data_analysis),
                "total_generated": len(generated_charts),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Auto visualization generation failed: {e}")
            return {
                "error": str(e),
                "charts": [],
                "recommendations": ["Try basic plotting methods manually"]
            }
    
    def _analyze_data_for_visualization(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze data characteristics for visualization decisions
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Data analysis results
        """
        analysis = {
            "shape": df.shape,
            "columns": list(df.columns),
            "numeric_columns": [],
            "categorical_columns": [],
            "datetime_columns": [],
            "boolean_columns": [],
            "high_cardinality_columns": [],
            "binary_columns": [],
            "missing_data": {},
            "data_ranges": {},
            "correlations": None
        }
        
        # Analyze each column
        for col in df.columns:
            try:
                dtype = df[col].dtype
                unique_count = df[col].nunique()
                total_count = len(df[col])
                
                # Categorize columns
                if pd.api.types.is_numeric_dtype(dtype):
                    analysis["numeric_columns"].append(col)
                    
                    # Calculate data ranges
                    analysis["data_ranges"][col] = {
                        "min": float(df[col].min()),
                        "max": float(df[col].max()),
                        "mean": float(df[col].mean()),
                        "std": float(df[col].std())
                    }
                    
                elif pd.api.types.is_datetime64_any_dtype(dtype):
                    analysis["datetime_columns"].append(col)
                    
                elif pd.api.types.is_bool_dtype(dtype):
                    analysis["boolean_columns"].append(col)
                    
                else:
                    analysis["categorical_columns"].append(col)
                    
                    # Check cardinality
                    if unique_count > 20:
                        analysis["high_cardinality_columns"].append(col)
                    elif unique_count == 2:
                        analysis["binary_columns"].append(col)
                
                # Missing data analysis
                missing_count = df[col].isnull().sum()
                if missing_count > 0:
                    analysis["missing_data"][col] = {
                        "count": int(missing_count),
                        "percentage": float(missing_count / total_count * 100)
                    }
                    
            except Exception as e:
                self.logger.warning(f"Error analyzing column {col}: {e}")
                continue
        
        # Calculate correlations for numeric data
        if len(analysis["numeric_columns"]) > 1:
            try:
                numeric_df = df[analysis["numeric_columns"]]
                analysis["correlations"] = numeric_df.corr().to_dict()
            except:
                pass
        
        return analysis
    
    def _recommend_chart_types(self, 
                             data_analysis: Dict[str, Any],
                             query: str,
                             intent: str) -> List[Dict[str, Any]]:
        """
        Recommend appropriate chart types based on data and context
        
        Args:
            data_analysis: Data analysis results
            query: User query
            intent: Analysis intent
            
        Returns:
            List of recommended chart configurations
        """
        recommendations = []
        
        numeric_cols = data_analysis["numeric_columns"]
        categorical_cols = data_analysis["categorical_columns"]
        datetime_cols = data_analysis["datetime_columns"]
        
        # Intent-based recommendations
        if "correlation" in intent.lower() or "relationship" in query.lower():
            if len(numeric_cols) >= 2:
                recommendations.append({
                    "type": "correlation",
                    "priority": 1,
                    "reason": "Correlation analysis for numeric variables"
                })
                
                recommendations.append({
                    "type": "scatter",
                    "columns": numeric_cols[:2],
                    "priority": 2,
                    "reason": "Scatter plot to visualize correlation"
                })
        
        elif "distribution" in query.lower() or intent == "exploration":
            # Distribution analysis
            for col in numeric_cols[:2]:
                recommendations.append({
                    "type": "histogram",
                    "columns": [col],
                    "priority": 1,
                    "reason": f"Distribution of {col}"
                })
        
        elif "comparison" in query.lower() or "compare" in query.lower():
            if categorical_cols and numeric_cols:
                recommendations.append({
                    "type": "box",
                    "x_column": categorical_cols[0],
                    "y_column": numeric_cols[0],
                    "priority": 1,
                    "reason": "Box plot for group comparison"
                })
        
        # Data-driven recommendations
        if len(numeric_cols) == 1 and len(categorical_cols) >= 1:
            recommendations.append({
                "type": "bar",
                "x_column": categorical_cols[0],
                "y_column": numeric_cols[0],
                "priority": 2,
                "reason": "Bar chart for categorical vs numeric"
            })
        
        elif len(numeric_cols) >= 2:
            recommendations.append({
                "type": "scatter",
                "columns": numeric_cols[:2],
                "priority": 2,
                "reason": "Scatter plot for numeric relationships"
            })
        
        elif len(categorical_cols) >= 1:
            for col in categorical_cols[:2]:
                if col not in data_analysis["high_cardinality_columns"]:
                    recommendations.append({
                        "type": "count",
                        "columns": [col],
                        "priority": 3,
                        "reason": f"Count plot for {col}"
                    })
        
        # Time series recommendations
        if datetime_cols and numeric_cols:
            recommendations.append({
                "type": "line",
                "x_column": datetime_cols[0],
                "y_column": numeric_cols[0],
                "priority": 1,
                "reason": "Time series visualization"
            })
        
        # Sort by priority and return
        recommendations.sort(key=lambda x: x.get("priority", 10))
        return recommendations
    
    def _generate_single_chart(self, 
                             df: pd.DataFrame,
                             chart_info: Dict[str, Any],
                             data_analysis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Generate a single chart based on configuration
        
        Args:
            df: DataFrame to plot
            chart_info: Chart configuration
            data_analysis: Data analysis context
            
        Returns:
            Chart result dictionary or None if failed
        """
        try:
            chart_type = chart_info["type"]
            
            if chart_type not in self.chart_templates:
                self.logger.warning(f"Unknown chart type: {chart_type}")
                return None
            
            # Create the chart
            fig, chart_data = self.chart_templates[chart_type](df, chart_info, data_analysis)
            
            if fig is None:
                return None
            
            # Convert to base64 for web display
            img_base64 = self._figure_to_base64(fig)
            
            # Generate code for reproducibility
            code = self._generate_chart_code(chart_type, chart_info, data_analysis)
            
            plt.close(fig)  # Clean up memory
            
            return {
                "type": chart_type,
                "image_base64": img_base64,
                "code": code,
                "description": chart_info.get("reason", f"{chart_type} visualization"),
                "columns_used": chart_info.get("columns", []),
                "insights": chart_data.get("insights", []),
                "metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "chart_config": chart_info
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error generating {chart_info.get('type', 'unknown')} chart: {e}")
            return None
    
    def _create_histogram(self, 
                         df: pd.DataFrame,
                         chart_info: Dict[str, Any],
                         data_analysis: Dict[str, Any]) -> Tuple[plt.Figure, Dict[str, Any]]:
        """Create histogram visualization"""
        
        columns = chart_info.get("columns", data_analysis["numeric_columns"][:1])
        if not columns:
            return None, {}
        
        col = columns[0]
        
        fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
        
        # Create histogram
        data = df[col].dropna()
        ax.hist(data, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        
        ax.set_title(f'Distribution of {col}')
        ax.set_xlabel(col)
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        mean_val = data.mean()
        ax.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
        ax.legend()
        
        plt.tight_layout()
        
        insights = [
            f"Mean value: {mean_val:.2f}",
            f"Standard deviation: {data.std():.2f}",
            f"Data points: {len(data):,}"
        ]
        
        return fig, {"insights": insights}
    
    def _create_scatter_plot(self, 
                           df: pd.DataFrame,
                           chart_info: Dict[str, Any],
                           data_analysis: Dict[str, Any]) -> Tuple[plt.Figure, Dict[str, Any]]:
        """Create scatter plot visualization"""
        
        columns = chart_info.get("columns", data_analysis["numeric_columns"][:2])
        if len(columns) < 2:
            return None, {}
        
        x_col, y_col = columns[0], columns[1]
        
        fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
        
        # Create scatter plot
        ax.scatter(df[x_col], df[y_col], alpha=0.6, color='coral')
        
        ax.set_title(f'{y_col} vs {x_col}')
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.grid(True, alpha=0.3)
        
        # Add trend line if correlation is strong
        correlation = df[x_col].corr(df[y_col])
        if abs(correlation) > 0.5:
            z = np.polyfit(df[x_col].dropna(), df[y_col].dropna(), 1)
            p = np.poly1d(z)
            ax.plot(df[x_col], p(df[x_col]), "r--", alpha=0.8, label=f'Trend (r={correlation:.2f})')
            ax.legend()
        
        plt.tight_layout()
        
        insights = [
            f"Correlation: {correlation:.3f}",
            f"Data points: {len(df):,}",
            "Strong relationship" if abs(correlation) > 0.7 else "Weak to moderate relationship"
        ]
        
        return fig, {"insights": insights}
    
    def _create_bar_chart(self, 
                         df: pd.DataFrame,
                         chart_info: Dict[str, Any],
                         data_analysis: Dict[str, Any]) -> Tuple[plt.Figure, Dict[str, Any]]:
        """Create bar chart visualization"""
        
        x_col = chart_info.get("x_column")
        y_col = chart_info.get("y_column")
        
        if not x_col or not y_col:
            return None, {}
        
        fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
        
        # Group and aggregate data
        grouped_data = df.groupby(x_col)[y_col].mean().sort_values(ascending=False)
        
        # Limit to top 10 categories for readability
        if len(grouped_data) > 10:
            grouped_data = grouped_data.head(10)
        
        grouped_data.plot(kind='bar', ax=ax, color='lightgreen', edgecolor='black')
        
        ax.set_title(f'Average {y_col} by {x_col}')
        ax.set_xlabel(x_col)
        ax.set_ylabel(f'Average {y_col}')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        insights = [
            f"Categories shown: {len(grouped_data)}",
            f"Highest value: {grouped_data.max():.2f}",
            f"Lowest value: {grouped_data.min():.2f}"
        ]
        
        return fig, {"insights": insights}
    
    def _create_correlation_matrix(self, 
                                 df: pd.DataFrame,
                                 chart_info: Dict[str, Any],
                                 data_analysis: Dict[str, Any]) -> Tuple[plt.Figure, Dict[str, Any]]:
        """Create correlation matrix heatmap"""
        
        numeric_cols = data_analysis["numeric_columns"]
        if len(numeric_cols) < 2:
            return None, {}
        
        fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
        
        # Calculate correlation matrix
        corr_matrix = df[numeric_cols].corr()
        
        # Create heatmap
        sns.heatmap(
            corr_matrix, 
            annot=True, 
            cmap='coolwarm', 
            center=0,
            square=True,
            fmt='.2f',
            ax=ax
        )
        
        ax.set_title('Correlation Matrix')
        plt.tight_layout()
        
        # Find strongest correlations
        strong_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.5:
                    strong_corr.append(f"{corr_matrix.columns[i]} ↔ {corr_matrix.columns[j]}: {corr_val:.3f}")
        
        insights = [
            f"Variables analyzed: {len(numeric_cols)}",
            f"Strong correlations found: {len(strong_corr)}"
        ] + strong_corr[:3]  # Top 3 strong correlations
        
        return fig, {"insights": insights}
    
    def _create_box_plot(self, 
                        df: pd.DataFrame,
                        chart_info: Dict[str, Any],
                        data_analysis: Dict[str, Any]) -> Tuple[plt.Figure, Dict[str, Any]]:
        """Create box plot visualization"""
        
        x_col = chart_info.get("x_column")
        y_col = chart_info.get("y_column")
        
        if not x_col or not y_col:
            return None, {}
        
        fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
        
        # Create box plot
        sns.boxplot(data=df, x=x_col, y=y_col, ax=ax)
        
        ax.set_title(f'{y_col} Distribution by {x_col}')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Calculate insights
        group_stats = df.groupby(x_col)[y_col].agg(['mean', 'median', 'std'])
        
        insights = [
            f"Groups compared: {len(group_stats)}",
            f"Overall range: {df[y_col].min():.2f} to {df[y_col].max():.2f}",
            "Box plot shows median, quartiles, and outliers"
        ]
        
        return fig, {"insights": insights}
    
    # Additional chart creation methods...
    def _create_line_plot(self, df, chart_info, data_analysis):
        """Create line plot (placeholder)"""
        return None, {}
    
    def _create_violin_plot(self, df, chart_info, data_analysis):
        """Create violin plot (placeholder)"""
        return None, {}
    
    def _create_heatmap(self, df, chart_info, data_analysis):
        """Create heatmap (placeholder)"""
        return None, {}
    
    def _create_distribution_plot(self, df, chart_info, data_analysis):
        """Create distribution plot (placeholder)"""
        return None, {}
    
    def _create_count_plot(self, df, chart_info, data_analysis):
        """Create count plot (placeholder)"""
        return None, {}
    
    def _create_regression_plot(self, df, chart_info, data_analysis):
        """Create regression plot (placeholder)"""
        return None, {}
    
    def _create_pair_plot(self, df, chart_info, data_analysis):
        """Create pair plot (placeholder)"""
        return None, {}
    
    def _figure_to_base64(self, fig: plt.Figure) -> str:
        """Convert matplotlib figure to base64 string"""
        try:
            img_buffer = io.BytesIO()
            fig.savefig(img_buffer, format='png', bbox_inches='tight', dpi=self.dpi)
            img_buffer.seek(0)
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
            img_buffer.close()
            return img_base64
        except Exception as e:
            self.logger.error(f"Error converting figure to base64: {e}")
            return ""
    
    def _generate_chart_code(self, 
                           chart_type: str,
                           chart_info: Dict[str, Any],
                           data_analysis: Dict[str, Any]) -> str:
        """Generate reproducible code for the chart"""
        
        code_templates = {
            "histogram": "plt.hist(df['{col}'], bins=30)\nplt.title('Distribution of {col}')\nplt.show()",
            "scatter": "plt.scatter(df['{x_col}'], df['{y_col}'])\nplt.xlabel('{x_col}')\nplt.ylabel('{y_col}')\nplt.show()",
            "bar": "df.groupby('{x_col}')['{y_col}'].mean().plot(kind='bar')\nplt.show()",
            "correlation": "sns.heatmap(df.corr(), annot=True)\nplt.show()",
            "box": "sns.boxplot(data=df, x='{x_col}', y='{y_col}')\nplt.show()"
        }
        
        template = code_templates.get(chart_type, f"# {chart_type} chart\n# Code template not available")
        
        # Format template with actual column names
        try:
            if chart_type == "histogram":
                columns = chart_info.get("columns", data_analysis["numeric_columns"][:1])
                if columns:
                    template = template.format(col=columns[0])
            elif chart_type in ["scatter"]:
                columns = chart_info.get("columns", data_analysis["numeric_columns"][:2])
                if len(columns) >= 2:
                    template = template.format(x_col=columns[0], y_col=columns[1])
            elif chart_type in ["bar", "box"]:
                x_col = chart_info.get("x_column", "")
                y_col = chart_info.get("y_column", "")
                template = template.format(x_col=x_col, y_col=y_col)
        except:
            pass
        
        return template
    
    def _generate_visualization_recommendations(self, data_analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations for further visualization"""
        recommendations = []
        
        numeric_count = len(data_analysis["numeric_columns"])
        categorical_count = len(data_analysis["categorical_columns"])
        
        if numeric_count > 2:
            recommendations.append("Consider pair plots for exploring relationships between all numeric variables")
        
        if categorical_count > 0 and numeric_count > 0:
            recommendations.append("Try grouped analysis (e.g., boxplots) to compare numeric values across categories")
        
        if data_analysis["missing_data"]:
            recommendations.append("Visualize missing data patterns with missingno library")
        
        if numeric_count > 0:
            recommendations.append("Use violin plots to see distribution shapes and density")
        
        recommendations.append("Consider interactive plots with plotly for better exploration")
        
        return recommendations 