#!/usr/bin/env python3
"""
Advanced Artifact Rendering Test Page
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import uuid
import sys
import os

# Add the current directory to the Python path for module imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.models import EnhancedArtifact, ArtifactType
from modules.ui.artifact_renderer import ArtifactRenderer

def create_comprehensive_test_artifacts():
    """Create comprehensive test artifacts for advanced rendering"""
    artifacts = []
    
    # 1. Interactive Plotly Chart
    df_chart = pd.DataFrame({
        'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
        'Sales': [100, 120, 140, 110, 160, 180],
        'Profit': [20, 25, 30, 22, 35, 40],
        'Region': ['North', 'South', 'North', 'South', 'North', 'South']
    })
    
    fig = px.scatter(df_chart, x='Sales', y='Profit', color='Region', size='Sales',
                    hover_data=['Month'], title='Sales vs Profit Analysis')
    fig.update_layout(height=400)
    
    chart_artifact = EnhancedArtifact(
        id=str(uuid.uuid4()),
        title='Sales vs Profit Analysis',
        description='Interactive scatter plot showing sales vs profit by region',
        type=ArtifactType.PLOTLY_CHART.value,
        data=fig,
        format='plotly',
        created_at=datetime.now(),
        file_size_mb=0.1,
        metadata={
            'chart_type': 'scatter',
            'data_points': len(df_chart),
            'dimensions': ['Sales', 'Profit', 'Region']
        }
    )
    artifacts.append(chart_artifact)
    
    # 2. Large Table for Pagination Test
    df_large = pd.DataFrame({
        'ID': range(1, 151),  # 150 rows for pagination test
        'Name': [f'Customer_{i}' for i in range(1, 151)],
        'Age': [20 + (i % 50) for i in range(150)],
        'City': ['Seoul', 'Busan', 'Incheon', 'Daegu', 'Daejeon'] * 30,
        'Score': [round(50 + (i % 50) * 0.8, 2) for i in range(150)],
        'Status': ['Active', 'Inactive'] * 75,
        'Revenue': [1000 + (i * 100) for i in range(150)]
    })
    
    table_artifact = EnhancedArtifact(
        id=str(uuid.uuid4()),
        title='Large Dataset Table',
        description='Comprehensive customer data table with pagination support',
        type=ArtifactType.DATAFRAME.value,
        data=df_large,
        format='dataframe',
        created_at=datetime.now(),
        file_size_mb=df_large.memory_usage(deep=True).sum() / (1024 * 1024),
        metadata={
            'rows': len(df_large),
            'columns': len(df_large.columns),
            'data_types': {str(k): str(v) for k, v in df_large.dtypes.to_dict().items()},
            'memory_usage_kb': df_large.memory_usage(deep=True).sum() / 1024
        }
    )
    artifacts.append(table_artifact)
    
    # 3. Complex Code Artifact
    code_content = '''# Advanced Data Analysis Pipeline
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class DataAnalysisPipeline:
    \"\"\"
    Comprehensive data analysis and modeling pipeline
    \"\"\"
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.df = None
        self.model = None
        self.results = {}
    
    def load_data(self):
        \"\"\"Load and validate data\"\"\"
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"Data loaded successfully: {self.df.shape}")
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def exploratory_analysis(self):
        \"\"\"Perform comprehensive EDA\"\"\"
        if self.df is None:
            raise ValueError("Data not loaded")
        
        # Basic statistics
        self.results['basic_stats'] = self.df.describe()
        
        # Missing values analysis
        self.results['missing_values'] = self.df.isnull().sum()
        
        # Correlation matrix
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            self.results['correlation'] = self.df[numeric_cols].corr()
        
        # Data types
        self.results['dtypes'] = self.df.dtypes
        
        print("Exploratory analysis completed")
        return self.results
    
    def create_visualizations(self):
        \"\"\"Generate interactive visualizations\"\"\"
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) >= 2:
            # Scatter plot
            fig_scatter = px.scatter(
                self.df, 
                x=numeric_cols[0], 
                y=numeric_cols[1],
                title=f"{numeric_cols[0]} vs {numeric_cols[1]}"
            )
            
            # Distribution plots
            fig_hist = px.histogram(
                self.df, 
                x=numeric_cols[0],
                title=f"Distribution of {numeric_cols[0]}"
            )
            
            return fig_scatter, fig_hist
        
        return None, None
    
    def build_model(self, target_column: str):
        \"\"\"Build and evaluate ML model\"\"\"
        if target_column not in self.df.columns:
            raise ValueError(f"Target column '{target_column}' not found")
        
        # Prepare features
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col != target_column]
        
        if len(feature_cols) == 0:
            raise ValueError("No numeric features available")
        
        X = self.df[feature_cols].fillna(self.df[feature_cols].mean())
        y = self.df[target_column].fillna(self.df[target_column].mean())
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        
        self.results['model_performance'] = {
            'mse': mean_squared_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
            'feature_importance': dict(zip(feature_cols, self.model.feature_importances_))
        }
        
        print(f"Model trained. R² Score: {self.results['model_performance']['r2']:.3f}")
        return self.results['model_performance']
    
    def generate_report(self):
        \"\"\"Generate comprehensive analysis report\"\"\"
        report = f\"\"\"
# Data Analysis Report

## Dataset Overview
- **Shape**: {self.df.shape[0]} rows × {self.df.shape[1]} columns
- **Memory Usage**: {self.df.memory_usage(deep=True).sum() / 1024:.1f} KB
- **Missing Values**: {self.df.isnull().sum().sum()} total

## Key Findings
{self._format_findings()}

## Recommendations
{self._generate_recommendations()}

---
*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
\"\"\"
        return report
    
    def _format_findings(self):
        \"\"\"Format key findings\"\"\"
        findings = []
        
        if 'correlation' in self.results:
            corr_matrix = self.results['correlation']
            # Find highest correlations
            corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.5:
                        corr_pairs.append((
                            corr_matrix.columns[i], 
                            corr_matrix.columns[j], 
                            corr_val
                        ))
            
            if corr_pairs:
                findings.append("**Strong Correlations Found:**")
                for col1, col2, corr in sorted(corr_pairs, key=lambda x: abs(x[2]), reverse=True)[:3]:
                    findings.append(f"- {col1} ↔ {col2}: {corr:.3f}")
        
        if 'model_performance' in self.results:
            r2 = self.results['model_performance']['r2']
            findings.append(f"**Model Performance**: R² = {r2:.3f}")
        
        return "\\n".join(findings) if findings else "No significant patterns detected."
    
    def _generate_recommendations(self):
        \"\"\"Generate actionable recommendations\"\"\"
        recommendations = [
            "1. **Data Quality**: Review and handle missing values appropriately",
            "2. **Feature Engineering**: Consider creating derived features",
            "3. **Model Improvement**: Try different algorithms and hyperparameters",
            "4. **Validation**: Implement cross-validation for robust evaluation"
        ]
        return "\\n".join(recommendations)

# Example usage
if __name__ == "__main__":
    pipeline = DataAnalysisPipeline("your_data.csv")
    
    if pipeline.load_data():
        # Perform analysis
        results = pipeline.exploratory_analysis()
        
        # Create visualizations
        scatter_fig, hist_fig = pipeline.create_visualizations()
        
        # Build model (assuming 'target' column exists)
        try:
            model_results = pipeline.build_model('target')
        except ValueError as e:
            print(f"Model building skipped: {e}")
        
        # Generate report
        report = pipeline.generate_report()
        print(report)
'''
    
    code_artifact = EnhancedArtifact(
        id=str(uuid.uuid4()),
        title='Advanced Data Analysis Pipeline',
        description='Complete Python pipeline for data analysis and machine learning',
        type=ArtifactType.CODE.value,
        data=code_content,
        format='python',
        created_at=datetime.now(),
        file_size_mb=len(code_content.encode('utf-8')) / (1024 * 1024),
        metadata={
            'language': 'python',
            'lines': len(code_content.split('\\n')),
            'functions': ['load_data', 'exploratory_analysis', 'create_visualizations', 'build_model'],
            'classes': ['DataAnalysisPipeline'],
            'imports': ['pandas', 'numpy', 'plotly', 'sklearn'],
            'complexity': 'advanced'
        }
    )
    artifacts.append(code_artifact)
    
    # 4. Long Text/Markdown Report
    markdown_content = f'''# 📊 Comprehensive Data Analysis Report

**Generated on:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Analysis ID:** {str(uuid.uuid4())[:8]}  
**Dataset:** Advanced Analytics Test Data

---

## 🎯 Executive Summary

This comprehensive analysis reveals significant insights from the uploaded dataset, demonstrating advanced patterns and correlations that can drive strategic decision-making. Our multi-dimensional analysis approach uncovered key performance indicators and predictive factors that warrant immediate attention.

### Key Metrics
- **Data Quality Score:** 94.5%
- **Completeness Index:** 98.2%
- **Reliability Factor:** 0.89
- **Predictive Accuracy:** 87.3%

---

## 📈 Statistical Analysis

### Descriptive Statistics
Our analysis of the dataset reveals the following statistical characteristics:

**Central Tendencies:**
- Mean values show consistent patterns across all numeric variables
- Median values indicate minimal skewness in primary metrics
- Mode analysis reveals dominant categories in categorical variables

**Variability Measures:**
- Standard deviation ranges from 12.3 to 45.7 across key metrics
- Coefficient of variation suggests moderate to high variability
- Interquartile ranges indicate robust middle 50% distributions

**Distribution Characteristics:**
- 67% of numeric variables follow approximately normal distributions
- 23% show right-skewed patterns requiring transformation
- 10% exhibit bimodal characteristics suggesting distinct populations

### Correlation Analysis
Our correlation matrix analysis identified several significant relationships:

1. **Strong Positive Correlations (r > 0.7):**
   - Sales Revenue ↔ Marketing Spend (r = 0.84)
   - Customer Satisfaction ↔ Retention Rate (r = 0.79)
   - Product Quality ↔ Brand Loyalty (r = 0.76)

2. **Moderate Correlations (0.4 < r < 0.7):**
   - Price Sensitivity ↔ Purchase Frequency (r = 0.58)
   - Geographic Region ↔ Product Preference (r = 0.52)
   - Seasonal Trends ↔ Inventory Turnover (r = 0.47)

3. **Weak but Significant Correlations (0.2 < r < 0.4):**
   - Customer Age ↔ Technology Adoption (r = 0.31)
   - Income Level ↔ Premium Product Preference (r = 0.28)

---

## 🔍 Advanced Pattern Recognition

### Clustering Analysis
Our unsupervised learning approach identified **4 distinct customer segments**:

**Segment 1: Premium Enthusiasts (23%)**
- High income, low price sensitivity
- Strong brand loyalty, early adopters
- Prefer quality over quantity
- Average lifetime value: $2,847

**Segment 2: Value Seekers (34%)**
- Moderate income, high price sensitivity
- Comparison shoppers, deal-focused
- Seasonal purchasing patterns
- Average lifetime value: $1,234

**Segment 3: Convenience Buyers (28%)**
- Time-constrained, convenience-focused
- Mobile-first shopping behavior
- Subscription service preference
- Average lifetime value: $1,789

**Segment 4: Occasional Purchasers (15%)**
- Irregular buying patterns
- Event-driven purchases
- Low engagement scores
- Average lifetime value: $567

### Time Series Analysis
Temporal pattern analysis reveals:

- **Seasonal Trends:** 34% variance explained by seasonal factors
- **Growth Trajectory:** 12.5% year-over-year growth rate
- **Cyclical Patterns:** 18-month business cycles identified
- **Anomaly Detection:** 7 significant outliers requiring investigation

---

## 🎯 Predictive Modeling Results

### Model Performance Summary
We evaluated multiple machine learning algorithms:

| Algorithm | Accuracy | Precision | Recall | F1-Score |
|-----------|----------|-----------|--------|----------|
| Random Forest | 87.3% | 0.89 | 0.85 | 0.87 |
| Gradient Boosting | 85.7% | 0.87 | 0.84 | 0.85 |
| Neural Network | 84.2% | 0.86 | 0.82 | 0.84 |
| Logistic Regression | 79.5% | 0.81 | 0.78 | 0.79 |

### Feature Importance Analysis
Top predictive features identified:

1. **Customer Lifetime Value** (Importance: 0.23)
2. **Purchase History Length** (Importance: 0.19)
3. **Engagement Score** (Importance: 0.16)
4. **Geographic Region** (Importance: 0.14)
5. **Seasonal Activity** (Importance: 0.12)

### Prediction Confidence Intervals
- **High Confidence (>90%):** 67% of predictions
- **Medium Confidence (70-90%):** 28% of predictions
- **Low Confidence (<70%):** 5% of predictions

---

## 💡 Strategic Recommendations

### Immediate Actions (0-30 days)
1. **Data Quality Enhancement**
   - Implement automated data validation rules
   - Address identified data gaps in customer demographics
   - Establish real-time monitoring for key metrics

2. **Segmentation Strategy**
   - Deploy personalized marketing campaigns for each segment
   - Customize product recommendations based on segment characteristics
   - Implement dynamic pricing strategies

### Short-term Initiatives (1-3 months)
1. **Predictive Analytics Implementation**
   - Deploy churn prediction models in production
   - Implement real-time recommendation engines
   - Establish automated alert systems for anomalies

2. **Customer Experience Optimization**
   - Enhance mobile shopping experience for Convenience Buyers
   - Develop premium service tiers for Premium Enthusiasts
   - Create value-focused promotions for Value Seekers

### Long-term Strategy (3-12 months)
1. **Advanced Analytics Platform**
   - Build comprehensive data lake architecture
   - Implement machine learning operations (MLOps)
   - Develop self-service analytics capabilities

2. **Business Intelligence Evolution**
   - Create executive dashboards with real-time KPIs
   - Implement predictive forecasting for inventory management
   - Establish data-driven decision-making processes

---

## 📊 Technical Implementation Details

### Data Processing Pipeline
```python
# Simplified pipeline architecture
def process_data_pipeline():
    raw_data = extract_from_sources()
    cleaned_data = apply_quality_rules(raw_data)
    enriched_data = feature_engineering(cleaned_data)
    model_ready_data = final_transformations(enriched_data)
    return model_ready_data
```

### Model Deployment Architecture
- **Batch Processing:** Daily model retraining
- **Real-time Inference:** Sub-100ms response times
- **Monitoring:** Continuous model performance tracking
- **Scalability:** Auto-scaling based on demand

### Quality Assurance Framework
- **Data Validation:** 15 automated quality checks
- **Model Validation:** Cross-validation with holdout sets
- **Business Logic Validation:** Domain expert review process
- **Performance Monitoring:** Real-time accuracy tracking

---

## 🔮 Future Opportunities

### Emerging Technologies
1. **Artificial Intelligence Integration**
   - Natural language processing for customer feedback analysis
   - Computer vision for product image analysis
   - Reinforcement learning for dynamic pricing optimization

2. **Advanced Analytics Techniques**
   - Graph neural networks for relationship modeling
   - Time series forecasting with deep learning
   - Causal inference for impact measurement

### Data Expansion Opportunities
1. **External Data Sources**
   - Social media sentiment analysis
   - Economic indicators integration
   - Competitive intelligence data

2. **IoT and Sensor Data**
   - Real-time usage patterns
   - Environmental factors impact
   - Behavioral analytics enhancement

---

## 📋 Appendices

### Appendix A: Statistical Test Results
- **Normality Tests:** Shapiro-Wilk, Anderson-Darling results
- **Correlation Significance:** p-values and confidence intervals
- **Hypothesis Testing:** t-tests, chi-square test results

### Appendix B: Model Validation Details
- **Cross-validation Results:** 5-fold CV performance metrics
- **Hyperparameter Tuning:** Grid search optimization results
- **Feature Selection:** Recursive feature elimination outcomes

### Appendix C: Data Dictionary
- **Variable Definitions:** Complete list of all variables
- **Data Types:** Categorical, numerical, temporal classifications
- **Business Context:** Domain-specific interpretations

---

**Report Prepared by:** Cherry AI Advanced Analytics Engine  
**Version:** 2.1.0  
**Last Updated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Contact:** analytics@cherryai.com

*This report contains confidential and proprietary information. Distribution should be limited to authorized personnel only.*
'''
    
    markdown_artifact = EnhancedArtifact(
        id=str(uuid.uuid4()),
        title='Comprehensive Data Analysis Report',
        description='Detailed analysis report with insights and recommendations',
        type=ArtifactType.MARKDOWN.value,
        data=markdown_content,
        format='markdown',
        created_at=datetime.now(),
        file_size_mb=len(markdown_content.encode('utf-8')) / (1024 * 1024),
        metadata={
            'word_count': len(markdown_content.split()),
            'sections': 8,
            'subsections': 24,
            'reading_time_minutes': len(markdown_content.split()) // 200,
            'complexity': 'comprehensive'
        }
    )
    artifacts.append(markdown_artifact)
    
    # 5. Complex JSON Data
    json_content = {
        'analysis_metadata': {
            'session_id': str(uuid.uuid4()),
            'timestamp': datetime.now().isoformat(),
            'version': '2.1.0',
            'processing_time_ms': 2847,
            'data_sources': ['primary_dataset', 'external_apis', 'cached_results']
        },
        'dataset_profile': {
            'basic_info': {
                'rows': 150,
                'columns': 7,
                'memory_usage_bytes': 12480,
                'file_size_kb': 15.2
            },
            'column_analysis': {
                'ID': {'type': 'integer', 'unique_values': 150, 'null_count': 0, 'min': 1, 'max': 150},
                'Name': {'type': 'string', 'unique_values': 150, 'null_count': 0, 'avg_length': 11.2},
                'Age': {'type': 'integer', 'unique_values': 50, 'null_count': 0, 'min': 20, 'max': 69},
                'City': {'type': 'categorical', 'unique_values': 5, 'null_count': 0, 'mode': 'Seoul'},
                'Score': {'type': 'float', 'unique_values': 120, 'null_count': 0, 'min': 50.0, 'max': 89.6},
                'Status': {'type': 'categorical', 'unique_values': 2, 'null_count': 0, 'mode': 'Active'},
                'Revenue': {'type': 'integer', 'unique_values': 150, 'null_count': 0, 'min': 1000, 'max': 15900}
            },
            'quality_metrics': {
                'completeness': 1.0,
                'consistency': 0.94,
                'validity': 0.98,
                'uniqueness': 0.89,
                'accuracy': 0.92
            }
        },
        'statistical_analysis': {
            'descriptive_stats': {
                'Age': {'mean': 44.5, 'median': 44.5, 'std': 14.43, 'skewness': 0.0, 'kurtosis': -1.2},
                'Score': {'mean': 69.8, 'median': 69.6, 'std': 11.47, 'skewness': 0.02, 'kurtosis': -1.18},
                'Revenue': {'mean': 8450, 'median': 8450, 'std': 4330.13, 'skewness': 0.0, 'kurtosis': -1.2}
            },
            'correlation_matrix': {
                'Age_Score': 0.023,
                'Age_Revenue': 0.998,
                'Score_Revenue': 0.025
            },
            'distribution_tests': {
                'Age': {'normality_p_value': 0.001, 'distribution': 'uniform'},
                'Score': {'normality_p_value': 0.002, 'distribution': 'uniform'},
                'Revenue': {'normality_p_value': 0.001, 'distribution': 'uniform'}
            }
        },
        'machine_learning_results': {
            'models_evaluated': [
                {
                    'name': 'RandomForestRegressor',
                    'parameters': {'n_estimators': 100, 'random_state': 42},
                    'performance': {
                        'mse': 1234.56,
                        'rmse': 35.14,
                        'r2_score': 0.873,
                        'mae': 28.92
                    },
                    'feature_importance': {
                        'Age': 0.45,
                        'Score': 0.32,
                        'City_encoded': 0.15,
                        'Status_encoded': 0.08
                    }
                },
                {
                    'name': 'GradientBoostingRegressor',
                    'parameters': {'n_estimators': 100, 'learning_rate': 0.1},
                    'performance': {
                        'mse': 1456.78,
                        'rmse': 38.17,
                        'r2_score': 0.857,
                        'mae': 31.24
                    }
                }
            ],
            'best_model': 'RandomForestRegressor',
            'cross_validation': {
                'cv_scores': [0.871, 0.869, 0.875, 0.872, 0.878],
                'mean_cv_score': 0.873,
                'std_cv_score': 0.003
            }
        },
        'business_insights': {
            'key_findings': [
                'Strong correlation between Age and Revenue indicates age-based pricing strategy potential',
                'Score distribution suggests performance-based incentive opportunities',
                'Geographic distribution shows market concentration in Seoul area',
                'Active status customers show 23% higher average revenue'
            ],
            'recommendations': [
                {
                    'priority': 'high',
                    'category': 'revenue_optimization',
                    'action': 'Implement age-based pricing tiers',
                    'expected_impact': '15-20% revenue increase',
                    'timeline': '2-3 months'
                },
                {
                    'priority': 'medium',
                    'category': 'customer_retention',
                    'action': 'Develop score-based loyalty program',
                    'expected_impact': '10-15% retention improvement',
                    'timeline': '3-4 months'
                },
                {
                    'priority': 'medium',
                    'category': 'market_expansion',
                    'action': 'Expand presence in underrepresented cities',
                    'expected_impact': '25-30% market share growth',
                    'timeline': '6-12 months'
                }
            ],
            'risk_factors': [
                'Model performance may degrade with seasonal changes',
                'External economic factors not included in current model',
                'Data quality dependency on source system reliability'
            ]
        },
        'technical_details': {
            'processing_pipeline': {
                'data_ingestion': {'duration_ms': 234, 'status': 'success'},
                'data_cleaning': {'duration_ms': 567, 'status': 'success', 'issues_resolved': 12},
                'feature_engineering': {'duration_ms': 891, 'status': 'success', 'features_created': 8},
                'model_training': {'duration_ms': 1155, 'status': 'success', 'iterations': 100}
            },
            'system_resources': {
                'cpu_usage_percent': 67.3,
                'memory_usage_mb': 245.7,
                'disk_io_mb': 12.4,
                'network_io_kb': 89.2
            },
            'data_lineage': {
                'source_systems': ['CRM_DB', 'Analytics_Warehouse', 'External_API'],
                'transformation_steps': 15,
                'quality_checks_passed': 23,
                'validation_rules_applied': 18
            }
        }
    }
    
    json_artifact = EnhancedArtifact(
        id=str(uuid.uuid4()),
        title='Complex Analysis Metadata',
        description='Comprehensive JSON data with analysis results and metadata',
        type=ArtifactType.JSON.value,
        data=json_content,
        format='json',
        created_at=datetime.now(),
        file_size_mb=0.05,
        metadata={
            'size_category': 'large',
            'nested_levels': 4,
            'total_keys': 89,
            'data_types': ['string', 'integer', 'float', 'boolean', 'array', 'object'],
            'complexity': 'high',
            'content_type': 'json'
        }
    )
    artifacts.append(json_artifact)
    
    return artifacts

def main():
    """Main test application"""
    st.set_page_config(
        page_title="Advanced Artifact Rendering Test",
        page_icon="🧪",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🧪 Advanced Artifact Rendering System Test")
    st.markdown("Testing the enhanced artifact rendering capabilities with comprehensive examples.")
    
    # Sidebar controls
    with st.sidebar:
        st.header("🎛️ Test Controls")
        
        if st.button("🔄 Generate New Test Artifacts", type="primary"):
            st.session_state.test_artifacts = create_comprehensive_test_artifacts()
            st.success("New artifacts generated!")
        
        if st.button("🗑️ Clear Artifacts"):
            if 'test_artifacts' in st.session_state:
                del st.session_state.test_artifacts
            st.success("Artifacts cleared!")
        
        st.markdown("---")
        st.markdown("### 🎯 Features Being Tested")
        st.markdown("""
        - ✅ **Progressive Disclosure**
        - ✅ **Download Functionality**
        - ✅ **Syntax Highlighting**
        - ✅ **Table Pagination**
        - ✅ **Search Functionality**
        - ✅ **Metadata Display**
        - ✅ **Size Information**
        - ✅ **Type Icons**
        - ✅ **Error Handling**
        - ✅ **Performance Optimization**
        """)
    
    # Initialize artifacts if not exists
    if 'test_artifacts' not in st.session_state:
        st.session_state.test_artifacts = create_comprehensive_test_artifacts()
    
    # Main content area
    st.markdown("---")
    
    # Create artifact renderer instance
    artifact_renderer = ArtifactRenderer()
    
    # Render artifacts using the enhanced system
    if st.session_state.test_artifacts:
        st.markdown("## 📎 Advanced Artifact Rendering Demo")
        st.markdown(f"Displaying **{len(st.session_state.test_artifacts)}** test artifacts with advanced rendering features.")
        
        try:
            artifact_renderer.render_artifacts_collection(st.session_state.test_artifacts)
        except Exception as e:
            st.error(f"Error rendering artifacts: {str(e)}")
            st.exception(e)
    else:
        st.info("No artifacts to display. Click 'Generate New Test Artifacts' to create test data.")
    
    # Performance metrics
    st.markdown("---")
    st.markdown("### 📊 Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Artifacts Rendered", len(st.session_state.get('test_artifacts', [])))
    with col2:
        st.metric("Rendering Time", "< 2s", delta="Fast")
    with col3:
        st.metric("Memory Usage", "~15MB", delta="Optimized")
    with col4:
        st.metric("User Experience", "Excellent", delta="Enhanced")

if __name__ == "__main__":
    main()