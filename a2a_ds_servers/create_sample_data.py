# Create Sample Data for A2A Data Science Servers Testing
# Generates various sample datasets for comprehensive testing

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def create_sample_datasets():
    """Create various sample datasets for testing."""
    
    # Ensure directory exists
    os.makedirs("artifacts/data/shared_dataframes", exist_ok=True)
    
    # 1. Titanic-like dataset
    np.random.seed(42)
    n_passengers = 891
    
    titanic_data = pd.DataFrame({
        'PassengerId': range(1, n_passengers + 1),
        'Survived': np.random.choice([0, 1], n_passengers, p=[0.6, 0.4]),
        'Pclass': np.random.choice([1, 2, 3], n_passengers, p=[0.2, 0.3, 0.5]),
        'Sex': np.random.choice(['male', 'female'], n_passengers, p=[0.65, 0.35]),
        'Age': np.random.normal(30, 12, n_passengers).clip(0, 80),
        'SibSp': np.random.poisson(0.5, n_passengers),
        'Parch': np.random.poisson(0.4, n_passengers),
        'Fare': np.random.exponential(32, n_passengers),
        'Embarked': np.random.choice(['C', 'Q', 'S'], n_passengers, p=[0.2, 0.1, 0.7])
    })
    
    # Add some missing values
    titanic_data.loc[np.random.choice(titanic_data.index, 177, replace=False), 'Age'] = np.nan
    titanic_data.loc[np.random.choice(titanic_data.index, 2, replace=False), 'Embarked'] = np.nan
    
    titanic_data.to_csv("artifacts/data/shared_dataframes/titanic.csv", index=False)
    titanic_data.to_pickle("artifacts/data/shared_dataframes/titanic.csv.pkl")
    
    # 2. Sales dataset
    n_sales = 1000
    start_date = datetime(2023, 1, 1)
    
    sales_data = pd.DataFrame({
        'SaleID': range(1, n_sales + 1),
        'Date': [start_date + timedelta(days=np.random.randint(0, 365)) for _ in range(n_sales)],
        'Product': np.random.choice(['Product_A', 'Product_B', 'Product_C', 'Product_D'], n_sales),
        'Region': np.random.choice(['North', 'South', 'East', 'West'], n_sales),
        'Salesperson': np.random.choice([f'Sales_{i}' for i in range(1, 21)], n_sales),
        'Quantity': np.random.randint(1, 100, n_sales),
        'UnitPrice': np.random.normal(50, 15, n_sales).clip(10, 200),
        'Discount': np.random.uniform(0, 0.3, n_sales)
    })
    
    sales_data['TotalRevenue'] = sales_data['Quantity'] * sales_data['UnitPrice'] * (1 - sales_data['Discount'])
    
    sales_data.to_csv("artifacts/data/shared_dataframes/sales_data.csv", index=False)
    
    # 3. Employee dataset
    n_employees = 500
    
    employee_data = pd.DataFrame({
        'EmployeeID': range(1, n_employees + 1),
        'Name': [f'Employee_{i}' for i in range(1, n_employees + 1)],
        'Department': np.random.choice(['Engineering', 'Sales', 'Marketing', 'HR', 'Finance'], n_employees),
        'Position': np.random.choice(['Junior', 'Senior', 'Lead', 'Manager', 'Director'], n_employees),
        'Salary': np.random.normal(75000, 25000, n_employees).clip(40000, 200000),
        'Experience': np.random.randint(0, 20, n_employees),
        'Performance': np.random.normal(3.5, 0.8, n_employees).clip(1, 5),
        'HireDate': [start_date + timedelta(days=np.random.randint(-1825, 0)) for _ in range(n_employees)]
    })
    
    employee_data.to_csv("artifacts/data/shared_dataframes/employee_data.csv", index=False)
    
    # 4. Simple sample data for general testing
    simple_data = pd.DataFrame({
        'x': np.random.normal(0, 1, 100),
        'y': np.random.normal(0, 1, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100),
        'value': np.random.uniform(0, 100, 100)
    })
    
    simple_data.to_csv("artifacts/data/shared_dataframes/sample_data.csv", index=False)
    
    # 5. Time series data
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    
    timeseries_data = pd.DataFrame({
        'Date': dates,
        'Value1': np.random.normal(100, 10, len(dates)) + np.sin(np.arange(len(dates)) * 2 * np.pi / 365) * 20,
        'Value2': np.random.normal(50, 5, len(dates)) + np.cos(np.arange(len(dates)) * 2 * np.pi / 365) * 10,
        'Category': np.random.choice(['Type1', 'Type2', 'Type3'], len(dates))
    })
    
    timeseries_data.to_csv("artifacts/data/shared_dataframes/timeseries_data.csv", index=False)
    
    print("✅ Sample datasets created successfully:")
    print(f"  📁 titanic.csv ({len(titanic_data)} rows)")
    print(f"  📁 sales_data.csv ({len(sales_data)} rows)")
    print(f"  📁 employee_data.csv ({len(employee_data)} rows)")
    print(f"  📁 sample_data.csv ({len(simple_data)} rows)")
    print(f"  📁 timeseries_data.csv ({len(timeseries_data)} rows)")
    
    return {
        "titanic": titanic_data,
        "sales": sales_data,
        "employee": employee_data,
        "sample": simple_data,
        "timeseries": timeseries_data
    }

if __name__ == "__main__":
    print("🗃️ Creating sample datasets for A2A Data Science Servers...")
    datasets = create_sample_datasets()
    print("🎉 All sample datasets created successfully!") 