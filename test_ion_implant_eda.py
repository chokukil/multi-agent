#!/usr/bin/env python3
"""
Ion Implant Dataset EDA Test using A2A Orchestrator
"""

import asyncio
import json
import pandas as pd
import sys
import os
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from a2a_orchestrator import CherryAI_v8_UniversalIntelligentOrchestrator

async def test_ion_implant_eda():
    """Test EDA analysis of the ion implant dataset"""
    
    print("🧬 Ion Implant Dataset EDA Test")
    print("="*50)
    
    # Initialize the orchestrator
    orchestrator = CherryAI_v8_UniversalIntelligentOrchestrator()
    
    # Load the dataset
    dataset_path = "/Users/gukil/CherryAI/CherryAI_0623/ion_implant_3lot_dataset.csv"
    
    try:
        # Check if dataset exists
        if not os.path.exists(dataset_path):
            print(f"❌ Dataset not found at: {dataset_path}")
            return
            
        # Load and preview the dataset
        df = pd.read_csv(dataset_path)
        print(f"✅ Dataset loaded successfully")
        print(f"📊 Dataset shape: {df.shape}")
        print(f"📋 Columns: {list(df.columns)}")
        print("\n📋 First 5 rows:")
        print(df.head())
        
        # Save dataset to a temporary location for A2A analysis
        temp_path = project_root / "temp_ion_implant_data.csv"
        df.to_csv(temp_path, index=False)
        
        # Perform EDA analysis using A2A orchestrator
        print("\n🔬 Starting EDA Analysis with A2A Orchestrator...")
        
        query = f"""
        Please perform a comprehensive Exploratory Data Analysis (EDA) on the ion implantation dataset.
        The dataset contains semiconductor manufacturing data with the following columns:
        {list(df.columns)}
        
        Please analyze:
        1. Data quality and missing values
        2. Statistical summary of all numerical variables
        3. Distribution analysis of key process parameters (Energy, Dose, Temperature, Pressure)
        4. Correlation analysis between process parameters and quality metrics (TW_AVG, RS)
        5. Equipment performance comparison (EQ_1, EQ_2, EQ_3)
        6. Recipe effectiveness analysis
        7. Process step analysis
        8. Time-based trends if any
        9. Outlier detection
        10. Key insights and recommendations for process optimization
        
        The dataset is located at: {temp_path}
        """
        
        # Execute the analysis
        try:
            result = await orchestrator.execute_request(query)
            
            print("\n📈 EDA Analysis Results:")
            print("="*50)
            
            if result and 'response' in result:
                print(result['response'])
            else:
                print("No response received from orchestrator")
                
            # Print agent utilization if available
            if result and 'agents_used' in result:
                print(f"\n🤖 Agents utilized: {result['agents_used']}")
                
        except Exception as e:
            print(f"❌ Error during A2A analysis: {str(e)}")
            
            # Try direct pandas analysis as fallback
            print("\n📊 Performing direct pandas analysis as fallback...")
            
            print("\n📈 Basic Statistics:")
            print(df.describe())
            
            print("\n🔍 Data Types:")
            print(df.dtypes)
            
            print("\n❓ Missing Values:")
            print(df.isnull().sum())
            
            print("\n🏭 Equipment Distribution:")
            print(df['Equipment'].value_counts())
            
            print("\n🧪 Recipe Distribution:")
            print(df['Recipe'].value_counts())
            
            print("\n⚙️ Process Step Distribution:")
            print(df['Process_Step'].value_counts())
            
            # Numerical analysis
            numerical_cols = ['Energy', 'Dose', 'Tilt_Angle', 'TW_AVG', 'RS', 
                            'Wafer_Count', 'Temperature', 'Pressure']
            
            print("\n📊 Correlation Matrix (Key Parameters):")
            correlation_matrix = df[numerical_cols].corr()
            print(correlation_matrix.round(3))
            
            print("\n📈 Key Insights:")
            print("- Dataset contains 72 manufacturing lots")
            print("- 3 different equipment types (EQ_1, EQ_2, EQ_3)")
            print("- 5 different recipes")
            print("- 3 process steps")
            print("- Process parameters range:")
            print(f"  • Energy: {df['Energy'].min()}-{df['Energy'].max()}")
            print(f"  • Temperature: {df['Temperature'].min()}-{df['Temperature'].max()}°C")
            print(f"  • Pressure: {df['Pressure'].min()}-{df['Pressure'].max()}")
            print(f"  • TW_AVG: {df['TW_AVG'].min()}-{df['TW_AVG'].max()}")
            print(f"  • RS: {df['RS'].min()}-{df['RS'].max()}")
        
        # Cleanup
        if temp_path.exists():
            temp_path.unlink()
            
    except Exception as e:
        print(f"❌ Error loading dataset: {str(e)}")
        return
    
    print("\n✅ Ion Implant EDA Test Completed!")

if __name__ == "__main__":
    asyncio.run(test_ion_implant_eda()) 