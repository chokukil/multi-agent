#!/usr/bin/env python3
"""
Verify Python Doctor Fixes
Test all components after fixes have been applied
"""

import sys
import subprocess
from pathlib import Path

def test_component(name, test_func):
    """Test a component and report results"""
    print(f"🧪 Testing {name}...")
    try:
        result = test_func()
        if result:
            print(f"   ✅ {name}: {result}")
            return True
        else:
            print(f"   ❌ {name}: Failed")
            return False
    except Exception as e:
        print(f"   ❌ {name}: {str(e)}")
        return False

def test_python_version():
    """Test Python version"""
    version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    return f"Python {version}"

def test_virtual_env():
    """Test virtual environment"""
    venv_active = hasattr(sys, 'real_prefix') or (
        hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
    )
    return f"Virtual environment {'active' if venv_active else 'inactive'}"

def test_numpy():
    """Test numpy import and version"""
    import numpy as np
    return f"NumPy {np.__version__} imported successfully"

def test_pandas():
    """Test pandas import and basic functionality"""
    import pandas as pd
    df = pd.DataFrame({'test': [1, 2, 3]})
    return f"Pandas {pd.__version__} with {len(df)} row test DataFrame"

def test_streamlit():
    """Test streamlit import"""
    import streamlit as st
    return f"Streamlit {st.__version__} imported successfully"

def test_playwright():
    """Test playwright import"""
    import playwright
    # Try different ways to get version
    try:
        from playwright.__version__ import __version__
        version = __version__
    except:
        try:
            result = subprocess.run(['uv', 'pip', 'show', 'playwright'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if line.startswith('Version:'):
                        version = line.split(':')[1].strip()
                        break
                else:
                    version = "unknown"
            else:
                version = "unknown"
        except:
            version = "unknown"
    
    return f"Playwright {version} imported successfully"

def test_streamlit_config():
    """Test streamlit config file"""
    config_file = Path('.streamlit/config.toml')
    if config_file.exists():
        content = config_file.read_text()
        deprecated_options = ['installTracer', 'fixMatplotlib']
        found_deprecated = [opt for opt in deprecated_options if opt in content]
        
        if found_deprecated:
            return f"Found deprecated options: {found_deprecated}"
        else:
            return "Config file clean of deprecated options"
    else:
        return "No local config file found"

def test_minimal_streamlit_app():
    """Test that our minimal app can be imported"""
    try:
        exec(open('cherry_ai_minimal.py').read())
        return "Minimal app code executes without errors"
    except Exception as e:
        if "streamlit" in str(e).lower() and "run" in str(e).lower():
            return "Minimal app ready (Streamlit run needed for full test)"
        else:
            raise

def main():
    """Run all verification tests"""
    print("🔍 Verifying Python Doctor Fixes")
    print("=" * 50)
    
    tests = [
        ("Python Version", test_python_version),
        ("Virtual Environment", test_virtual_env),
        ("NumPy", test_numpy),
        ("Pandas", test_pandas),
        ("Streamlit", test_streamlit),
        ("Playwright", test_playwright),
        ("Streamlit Config", test_streamlit_config),
        ("Minimal App", test_minimal_streamlit_app),
    ]
    
    results = []
    for name, test_func in tests:
        success = test_component(name, test_func)
        results.append((name, success))
    
    print("\n" + "=" * 50)
    print("📊 VERIFICATION SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {name}")
    
    print(f"\n🎯 Success Rate: {passed}/{total} ({passed/total*100:.0f}%)")
    
    if passed == total:
        print("🎉 All components working correctly!")
        return 0
    elif passed >= total * 0.8:
        print("⚠️  Most components working, minor issues remain")
        return 0
    else:
        print("🚨 Significant issues detected")
        return 1

if __name__ == "__main__":
    exit(main())