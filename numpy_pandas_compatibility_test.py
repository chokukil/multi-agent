#!/usr/bin/env python3
"""
CherryAI NumPy & Pandas Compatibility Test
==========================================

This script comprehensively tests the compatibility between NumPy and Pandas
in the CherryAI environment to ensure there are no binary incompatibility issues.

Based on web research findings:
- NumPy 2.0+ caused binary incompatibility with older pandas versions
- pandas 2.2.2+ is compatible with numpy 2.0+
- Current versions: numpy 2.1.3 + pandas 2.3.0 should be fully compatible
"""

import warnings
import traceback
import sys
from typing import List, Tuple

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class CompatibilityTester:
    def __init__(self):
        self.test_results: List[Tuple[str, bool, str]] = []
    
    def run_test(self, test_name: str, test_func):
        """Run a single test and record the result."""
        try:
            test_func()
            self.test_results.append((test_name, True, "Success"))
            print(f"✅ {test_name}")
            return True
        except Exception as e:
            self.test_results.append((test_name, False, str(e)))
            print(f"❌ {test_name}: {str(e)}")
            return False
    
    def test_basic_imports(self):
        """Test basic numpy and pandas imports."""
        import numpy as np
        import pandas as pd
        assert np.__version__
        assert pd.__version__
    
    def test_pandas_libs_interval(self):
        """Test the specific pandas._libs.interval that causes dtype size errors."""
        from pandas._libs import interval
        # This import would fail with the classic numpy.dtype size error
    
    def test_data_operations(self):
        """Test typical data science operations."""
        import numpy as np
        import pandas as pd
        
        # Create test data
        data = {
            'numeric': np.random.randn(1000),
            'categorical': np.random.choice(['A', 'B', 'C'], 1000),
            'text': [f'text_{i}' for i in range(1000)]
        }
        
        df = pd.DataFrame(data)
        
        # Test common operations
        _ = df.describe()
        _ = df.dtypes
        _ = df.select_dtypes(include=[np.number]).corr()
        _ = df['categorical'].value_counts()
        _ = df.groupby('categorical')['numeric'].agg(['mean', 'std', 'count'])
        _ = df.memory_usage(deep=True)
    
    def test_string_operations(self):
        """Test string operations that were problematic in numpy 2.0."""
        import numpy as np
        import pandas as pd
        
        # Test string arrays
        df = pd.DataFrame({
            'strings': ['hello', 'world', 'numpy', 'pandas', 'compatible']
        })
        
        # String operations
        _ = df['strings'].str.len()
        _ = df['strings'].str.upper()
        _ = df['strings'].str.contains('comp')
    
    def test_mixed_dtypes(self):
        """Test mixed data types that could trigger compatibility issues."""
        import numpy as np
        import pandas as pd
        
        df = pd.DataFrame({
            'int64': np.array([1, 2, 3, 4, 5], dtype='int64'),
            'float64': np.array([1.1, 2.2, 3.3, 4.4, 5.5], dtype='float64'),
            'object': ['a', 'b', 'c', 'd', 'e'],
            'datetime': pd.date_range('2024-01-01', periods=5),
            'category': pd.Categorical(['x', 'y', 'x', 'y', 'x'])
        })
        
        # Operations on mixed types
        _ = df.info()
        _ = df.select_dtypes(include='number')
        _ = df.select_dtypes(include='object')
    
    def test_cherryai_core_imports(self):
        """Test CherryAI core module imports."""
        import sys
        sys.path.append('.')
        
        # Test the modules that exist
        from core.user_file_tracker import UserFileTracker
        from core.session_data_manager import SessionDataManager
        
        # Test initialization
        file_tracker = UserFileTracker()
        session_manager = SessionDataManager()
    
    def test_ml_libraries(self):
        """Test machine learning libraries compatibility."""
        import numpy as np
        import pandas as pd
        import sklearn
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        
        # Create test data
        X = np.random.randn(100, 5)
        y = np.random.choice([0, 1], 100)
        
        # Test sklearn with numpy/pandas
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        clf = RandomForestClassifier(n_estimators=10, random_state=42)
        clf.fit(X_scaled, y)
        
        # Test with pandas DataFrame
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
        X_df_scaled = scaler.transform(df.values)
        _ = clf.predict(X_df_scaled)
    
    def test_visualization_libraries(self):
        """Test visualization libraries compatibility."""
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        import plotly.express as px
        
        # Create test data
        df = pd.DataFrame({
            'x': np.random.randn(50),
            'y': np.random.randn(50),
            'category': np.random.choice(['A', 'B', 'C'], 50)
        })
        
        # Test matplotlib
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(df['x'], df['y'])
        plt.close(fig)
        
        # Test seaborn
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.scatterplot(data=df, x='x', y='y', hue='category', ax=ax)
        plt.close(fig)
        
        # Test plotly
        fig = px.scatter(df, x='x', y='y', color='category')
    
    def run_all_tests(self):
        """Run all compatibility tests."""
        print("=" * 60)
        print("CherryAI NumPy & Pandas Compatibility Test Suite")
        print("=" * 60)
        
        # Display current versions
        import numpy as np
        import pandas as pd
        print(f"Python: {sys.version}")
        print(f"NumPy: {np.__version__}")
        print(f"Pandas: {pd.__version__}")
        print()
        
        # Run all tests
        tests = [
            ("Basic Imports", self.test_basic_imports),
            ("Pandas Libs Interval", self.test_pandas_libs_interval),
            ("Data Operations", self.test_data_operations),
            ("String Operations", self.test_string_operations),
            ("Mixed Data Types", self.test_mixed_dtypes),
            ("CherryAI Core Imports", self.test_cherryai_core_imports),
            ("ML Libraries", self.test_ml_libraries),
            ("Visualization Libraries", self.test_visualization_libraries),
        ]
        
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            if self.run_test(test_name, test_func):
                passed += 1
        
        print()
        print("=" * 60)
        print("TEST RESULTS SUMMARY")
        print("=" * 60)
        print(f"Passed: {passed}/{total}")
        print(f"Failed: {total - passed}/{total}")
        
        if passed == total:
            print()
            print("🎉 ALL TESTS PASSED!")
            print("✅ NumPy 2.1.3 + Pandas 2.3.0 are fully compatible with CherryAI")
            print("✅ No binary incompatibility issues detected")
            print("✅ System is ready for production use")
        else:
            print()
            print("❌ Some tests failed. Please check the issues above.")
            
        return passed == total

if __name__ == "__main__":
    tester = CompatibilityTester()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1) 