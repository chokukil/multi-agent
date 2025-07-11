#!/usr/bin/env python3
"""
Detailed Ion Implant Dataset Analysis with Visualizations
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

def analyze_ion_implant_dataset():
    """Perform comprehensive analysis of ion implant dataset"""
    
    print("🧬 Detailed Ion Implant Dataset Analysis")
    print("="*60)
    
    # Load dataset
    df = pd.read_csv("/Users/gukil/CherryAI/CherryAI_0623/ion_implant_3lot_dataset.csv")
    
    # Create output directory for plots
    import os
    os.makedirs("artifacts/plots", exist_ok=True)
    
    print(f"📊 Dataset Overview:")
    print(f"   • Shape: {df.shape}")
    print(f"   • No missing values: {df.isnull().sum().sum() == 0}")
    print(f"   • Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    # 1. Process Parameter Analysis
    print("\n📈 1. Process Parameter Distribution Analysis")
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 4, figsize=(20, 12))
    fig.suptitle('Ion Implant Process Parameters Distribution', fontsize=16, fontweight='bold')
    
    # Plot distributions for key parameters
    numerical_params = ['Energy', 'Dose', 'Temperature', 'Pressure', 'TW_AVG', 'RS', 'Tilt_Angle', 'Wafer_Count']
    
    for i, param in enumerate(numerical_params):
        row = i // 4
        col = i % 4
        
        # Histogram with KDE
        axes[row, col].hist(df[param], bins=15, alpha=0.7, density=True, color='skyblue', edgecolor='black')
        axes[row, col].set_title(f'{param} Distribution', fontweight='bold')
        axes[row, col].set_xlabel(param)
        axes[row, col].set_ylabel('Density')
        
        # Add statistics text
        mean_val = df[param].mean()
        std_val = df[param].std()
        axes[row, col].axvline(mean_val, color='red', linestyle='--', alpha=0.8, label=f'Mean: {mean_val:.1f}')
        axes[row, col].legend()
    
    plt.tight_layout()
    plt.savefig('artifacts/plots/process_parameters_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Equipment Performance Analysis
    print("\n🏭 2. Equipment Performance Analysis")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Equipment Performance Analysis', fontsize=16, fontweight='bold')
    
    # Equipment usage distribution
    equipment_counts = df['Equipment'].value_counts()
    axes[0, 0].pie(equipment_counts.values, labels=equipment_counts.index, autopct='%1.1f%%', 
                   colors=['lightblue', 'lightgreen', 'lightcoral'])
    axes[0, 0].set_title('Equipment Usage Distribution')
    
    # Performance by equipment (TW_AVG)
    df.boxplot(column='TW_AVG', by='Equipment', ax=axes[0, 1])
    axes[0, 1].set_title('TW_AVG Performance by Equipment')
    axes[0, 1].set_xlabel('Equipment')
    
    # Performance by equipment (RS)
    df.boxplot(column='RS', by='Equipment', ax=axes[1, 0])
    axes[1, 0].set_title('RS Performance by Equipment')
    axes[1, 0].set_xlabel('Equipment')
    
    # Temperature vs Equipment
    sns.violinplot(data=df, x='Equipment', y='Temperature', ax=axes[1, 1])
    axes[1, 1].set_title('Temperature Distribution by Equipment')
    
    plt.tight_layout()
    plt.savefig('artifacts/plots/equipment_performance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Recipe Effectiveness Analysis
    print("\n🧪 3. Recipe Effectiveness Analysis")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Recipe Effectiveness Analysis', fontsize=16, fontweight='bold')
    
    # Recipe usage
    recipe_counts = df['Recipe'].value_counts()
    axes[0, 0].bar(recipe_counts.index, recipe_counts.values, color='lightsteelblue', edgecolor='black')
    axes[0, 0].set_title('Recipe Usage Frequency')
    axes[0, 0].set_xlabel('Recipe')
    axes[0, 0].set_ylabel('Count')
    
    # TW_AVG by Recipe
    sns.boxplot(data=df, x='Recipe', y='TW_AVG', ax=axes[0, 1])
    axes[0, 1].set_title('TW_AVG Performance by Recipe')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # RS by Recipe
    sns.boxplot(data=df, x='Recipe', y='RS', ax=axes[1, 0])
    axes[1, 0].set_title('RS Performance by Recipe')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Recipe vs Process Step heatmap
    recipe_step_matrix = pd.crosstab(df['Recipe'], df['Process_Step'])
    sns.heatmap(recipe_step_matrix, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1])
    axes[1, 1].set_title('Recipe vs Process Step Matrix')
    
    plt.tight_layout()
    plt.savefig('artifacts/plots/recipe_effectiveness.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. Correlation Analysis
    print("\n🔗 4. Correlation Analysis")
    
    # Select numerical columns for correlation
    numerical_cols = ['Energy', 'Dose', 'Tilt_Angle', 'TW_AVG', 'RS', 'Wafer_Count', 'Temperature', 'Pressure']
    correlation_matrix = df[numerical_cols].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='RdBu_r', center=0, 
                square=True, fmt='.3f', cbar_kws={'shrink': 0.8})
    plt.title('Process Parameters Correlation Matrix', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('artifacts/plots/correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 5. Quality Metrics Analysis
    print("\n⭐ 5. Quality Metrics Analysis")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Quality Metrics Analysis', fontsize=16, fontweight='bold')
    
    # TW_AVG vs RS scatter plot
    scatter = axes[0, 0].scatter(df['TW_AVG'], df['RS'], c=df['Energy'], cmap='viridis', alpha=0.7, s=60)
    axes[0, 0].set_xlabel('TW_AVG')
    axes[0, 0].set_ylabel('RS')
    axes[0, 0].set_title('TW_AVG vs RS (colored by Energy)')
    plt.colorbar(scatter, ax=axes[0, 0], label='Energy')
    
    # Energy vs Temperature with quality metrics
    scatter2 = axes[0, 1].scatter(df['Energy'], df['Temperature'], c=df['TW_AVG'], 
                                  cmap='plasma', alpha=0.7, s=60)
    axes[0, 1].set_xlabel('Energy')
    axes[0, 1].set_ylabel('Temperature')
    axes[0, 1].set_title('Energy vs Temperature (colored by TW_AVG)')
    plt.colorbar(scatter2, ax=axes[0, 1], label='TW_AVG')
    
    # Dose vs Pressure
    scatter3 = axes[1, 0].scatter(df['Dose'], df['Pressure'], c=df['RS'], 
                                  cmap='coolwarm', alpha=0.7, s=60)
    axes[1, 0].set_xlabel('Dose')
    axes[1, 0].set_ylabel('Pressure')
    axes[1, 0].set_title('Dose vs Pressure (colored by RS)')
    plt.colorbar(scatter3, ax=axes[1, 0], label='RS')
    
    # Quality score distribution
    # Create a composite quality score
    df['Quality_Score'] = (df['TW_AVG'] / df['TW_AVG'].max() + df['RS'] / df['RS'].max()) / 2
    axes[1, 1].hist(df['Quality_Score'], bins=15, alpha=0.7, color='gold', edgecolor='black')
    axes[1, 1].set_xlabel('Normalized Quality Score')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Quality Score Distribution')
    
    plt.tight_layout()
    plt.savefig('artifacts/plots/quality_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 6. Statistical Summary
    print("\n📊 6. Statistical Summary")
    
    print("\n🏆 Top 5 Best Performing Lots (highest quality score):")
    top_lots = df.nlargest(5, 'Quality_Score')[['LOT_ID', 'Equipment', 'Recipe', 'TW_AVG', 'RS', 'Quality_Score']]
    print(top_lots.to_string(index=False))
    
    print("\n📉 Bottom 5 Performing Lots (lowest quality score):")
    bottom_lots = df.nsmallest(5, 'Quality_Score')[['LOT_ID', 'Equipment', 'Recipe', 'TW_AVG', 'RS', 'Quality_Score']]
    print(bottom_lots.to_string(index=False))
    
    print("\n🏭 Equipment Performance Summary:")
    equipment_performance = df.groupby('Equipment').agg({
        'TW_AVG': ['mean', 'std'],
        'RS': ['mean', 'std'],
        'Quality_Score': ['mean', 'std'],
        'Temperature': 'mean',
        'Pressure': 'mean'
    }).round(2)
    print(equipment_performance)
    
    print("\n🧪 Recipe Performance Summary:")
    recipe_performance = df.groupby('Recipe').agg({
        'TW_AVG': ['mean', 'std'],
        'RS': ['mean', 'std'],
        'Quality_Score': ['mean', 'std']
    }).round(2)
    print(recipe_performance)
    
    # 7. Key Findings and Recommendations
    print("\n🔍 7. Key Findings and Recommendations")
    print("="*50)
    
    # Equipment analysis
    eq_performance = df.groupby('Equipment')['Quality_Score'].mean().sort_values(ascending=False)
    best_equipment = eq_performance.index[0]
    worst_equipment = eq_performance.index[-1]
    
    # Recipe analysis
    recipe_performance_mean = df.groupby('Recipe')['Quality_Score'].mean().sort_values(ascending=False)
    best_recipe = recipe_performance_mean.index[0]
    worst_recipe = recipe_performance_mean.index[-1]
    
    # Correlation insights
    strong_correlations = correlation_matrix.abs().unstack().sort_values(ascending=False)
    strong_correlations = strong_correlations[strong_correlations < 1.0]  # Remove self-correlations
    
    print(f"✅ EQUIPMENT INSIGHTS:")
    print(f"   • Best performing equipment: {best_equipment} (avg quality: {eq_performance[best_equipment]:.3f})")
    print(f"   • Worst performing equipment: {worst_equipment} (avg quality: {eq_performance[worst_equipment]:.3f})")
    print(f"   • Equipment {best_equipment} shows {((eq_performance[best_equipment] - eq_performance[worst_equipment])/eq_performance[worst_equipment]*100):.1f}% better performance")
    
    print(f"\n🧪 RECIPE INSIGHTS:")
    print(f"   • Best performing recipe: {best_recipe} (avg quality: {recipe_performance_mean[best_recipe]:.3f})")
    print(f"   • Worst performing recipe: {worst_recipe} (avg quality: {recipe_performance_mean[worst_recipe]:.3f})")
    
    print(f"\n🔗 CORRELATION INSIGHTS:")
    print(f"   • Strongest positive correlation: {strong_correlations.index[0][0]} vs {strong_correlations.index[0][1]} ({strong_correlations.iloc[0]:.3f})")
    print(f"   • Temperature shows correlation with Dose ({correlation_matrix.loc['Temperature', 'Dose']:.3f})")
    
    print(f"\n💡 OPTIMIZATION RECOMMENDATIONS:")
    print(f"   1. Focus on {best_equipment} for critical lots")
    print(f"   2. Optimize {best_recipe} parameters for better yield")
    print(f"   3. Monitor Temperature-Dose relationship for process control")
    print(f"   4. Investigate {worst_equipment} maintenance and calibration")
    print(f"   5. Energy range optimization: Current range {df['Energy'].min()}-{df['Energy'].max()}, consider focusing on {df[df['Quality_Score'] > df['Quality_Score'].quantile(0.75)]['Energy'].mean():.0f}±{df[df['Quality_Score'] > df['Quality_Score'].quantile(0.75)]['Energy'].std():.0f}")
    
    print(f"\n📈 PROCESS CONTROL RECOMMENDATIONS:")
    print(f"   • Temperature control: ±{df['Temperature'].std():.1f}°C variation observed")
    print(f"   • Pressure control: ±{df['Pressure'].std():.1f} variation observed")
    print(f"   • Dose consistency: {df['Dose'].std():.0f} standard deviation")
    
    print("\n✅ Analysis completed! All plots saved to artifacts/plots/")

if __name__ == "__main__":
    analyze_ion_implant_dataset() 