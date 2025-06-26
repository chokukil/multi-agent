#!/usr/bin/env python3
"""
테스트용 데이터 생성 및 업로드 스크립트
"""

import pandas as pd
import numpy as np
import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(project_root, 'core'))

from data_manager import DataManager

def create_sample_data():
    """Create sample datasets for testing"""
    
    # Sample 1: Sales data
    np.random.seed(42)
    sales_data = pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=100, freq='D'),
        'product': np.random.choice(['A', 'B', 'C', 'D'], 100),
        'sales': np.random.randint(10, 100, 100),
        'revenue': np.random.uniform(100, 1000, 100),
        'region': np.random.choice(['North', 'South', 'East', 'West'], 100)
    })
    
    # Sample 2: Customer data
    customer_data = pd.DataFrame({
        'customer_id': range(1, 51),
        'age': np.random.randint(18, 80, 50),
        'gender': np.random.choice(['M', 'F'], 50),
        'income': np.random.randint(30000, 120000, 50),
        'satisfaction': np.random.uniform(1, 10, 50),
        'city': np.random.choice(['Seoul', 'Busan', 'Daegu', 'Incheon'], 50)
    })
    
    return sales_data, customer_data

def main():
    print("🔧 Creating test datasets...")
    
    # Create sample data
    sales_data, customer_data = create_sample_data()
    
    # Initialize DataManager
    dm = DataManager()
    
    # Upload datasets
    print("📤 Uploading datasets to DataManager...")
    
    sales_id = dm.add_dataframe("sales_data.csv", sales_data, "Test script")
    customer_id = dm.add_dataframe("customer_data.csv", customer_data, "Test script")
    
    print(f"✅ Sales data uploaded with ID: {sales_id}")
    print(f"✅ Customer data uploaded with ID: {customer_id}")
    
    # List available datasets
    available = dm.list_dataframes()
    print(f"📋 Available datasets: {available}")
    
    # Show dataset info
    print("\n📊 Dataset Information:")
    for df_id in available:
        info = dm.get_data_info(df_id)
        if info:
            print(f"  - {df_id}: {info['metadata']['shape']} rows x cols, created: {info['created_at']}")

if __name__ == "__main__":
    main() 