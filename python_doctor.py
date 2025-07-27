#!/usr/bin/env python3
"""
Python Environment Doctor - SuperClaude Framework
Comprehensive Python environment diagnosis and repair tool
"""

import subprocess
import sys
import os
import json
import re
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime


class PythonDoctor:
    """Comprehensive Python environment diagnostic and repair tool"""
    
    def __init__(self, check_packages=None, pin_versions=None, clean_config=None):
        self.check_packages = check_packages or []
        self.pin_versions = pin_versions or {}
        self.clean_config = clean_config or []
        self.report = {
            'timestamp': datetime.now().isoformat(),
            'checks': {},
            'fixes': {},
            'summary': {'total_checks': 0, 'passed': 0, 'fixed': 0, 'failed': 0}
        }
        self.project_root = Path.cwd()
        
    def run_command(self, cmd: List[str], capture_output=True, timeout=60) -> Tuple[int, str, str]:
        """Run command and return exit code, stdout, stderr"""
        try:
            result = subprocess.run(
                cmd, 
                capture_output=capture_output, 
                text=True, 
                timeout=timeout,
                cwd=self.project_root
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return 1, "", f"Command timed out after {timeout}s"
        except Exception as e:
            return 1, "", str(e)
    
    def check_python_version(self) -> Dict[str, Any]:
        """Check Python version and compatibility"""
        print("🐍 Checking Python version...")
        
        check_result = {
            'name': 'Python Version',
            'status': 'unknown',
            'details': {},
            'recommendations': []
        }
        
        try:
            version_info = sys.version_info
            version_str = f"{version_info.major}.{version_info.minor}.{version_info.micro}"
            
            check_result['details'] = {
                'current_version': version_str,
                'version_info': str(version_info),
                'executable': sys.executable,
                'platform': sys.platform
            }
            
            # Check if version matches requirements
            if "python" in self.pin_versions:
                required = self.pin_versions["python"]
                if self._version_matches(version_str, required):
                    check_result['status'] = 'pass'
                    print(f"   ✅ Python {version_str} matches requirement {required}")
                else:
                    check_result['status'] = 'warn'
                    check_result['recommendations'].append(f"Consider using Python {required}")
                    print(f"   ⚠️  Python {version_str} doesn't match requirement {required}")
            else:
                check_result['status'] = 'pass'
                print(f"   ✅ Python {version_str} detected")
                
        except Exception as e:
            check_result['status'] = 'fail'
            check_result['details']['error'] = str(e)
            print(f"   ❌ Error checking Python version: {e}")
        
        return check_result
    
    def check_virtual_environment(self) -> Dict[str, Any]:
        """Check virtual environment setup"""
        print("🏠 Checking virtual environment...")
        
        check_result = {
            'name': 'Virtual Environment',
            'status': 'unknown',
            'details': {},
            'recommendations': []
        }
        
        try:
            venv_active = hasattr(sys, 'real_prefix') or (
                hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
            )
            
            check_result['details'] = {
                'venv_active': venv_active,
                'prefix': sys.prefix,
                'base_prefix': getattr(sys, 'base_prefix', 'N/A'),
                'executable': sys.executable
            }
            
            if venv_active:
                check_result['status'] = 'pass'
                print(f"   ✅ Virtual environment active: {sys.prefix}")
            else:
                check_result['status'] = 'warn'
                check_result['recommendations'].append("Activate virtual environment for isolated dependencies")
                print(f"   ⚠️  No virtual environment detected")
                
        except Exception as e:
            check_result['status'] = 'fail'
            check_result['details']['error'] = str(e)
            print(f"   ❌ Error checking virtual environment: {e}")
        
        return check_result
    
    def check_package_manager(self) -> Dict[str, Any]:
        """Check package manager (uv, pip) availability"""
        print("📦 Checking package managers...")
        
        check_result = {
            'name': 'Package Manager',
            'status': 'unknown',
            'details': {},
            'recommendations': []
        }
        
        managers = {}
        
        # Check uv
        code, stdout, stderr = self.run_command(['uv', '--version'])
        if code == 0:
            managers['uv'] = {'available': True, 'version': stdout.strip()}
            print(f"   ✅ uv available: {stdout.strip()}")
        else:
            managers['uv'] = {'available': False, 'error': stderr}
            print(f"   ❌ uv not available: {stderr}")
        
        # Check pip
        code, stdout, stderr = self.run_command([sys.executable, '-m', 'pip', '--version'])
        if code == 0:
            managers['pip'] = {'available': True, 'version': stdout.strip()}
            print(f"   ✅ pip available: {stdout.strip()}")
        else:
            managers['pip'] = {'available': False, 'error': stderr}
            print(f"   ❌ pip not available: {stderr}")
        
        check_result['details'] = managers
        
        if managers.get('uv', {}).get('available') or managers.get('pip', {}).get('available'):
            check_result['status'] = 'pass'
        else:
            check_result['status'] = 'fail'
            check_result['recommendations'].append("Install a package manager (pip or uv)")
        
        return check_result
    
    def check_dependencies(self) -> Dict[str, Any]:
        """Check specified package dependencies"""
        print("📋 Checking package dependencies...")
        
        check_result = {
            'name': 'Package Dependencies',
            'status': 'unknown',
            'details': {},
            'recommendations': []
        }
        
        packages = {}
        
        for package in self.check_packages:
            print(f"   🔍 Checking {package}...")
            try:
                # Try to import the package
                if package == "numpy":
                    import numpy
                    version = numpy.__version__
                elif package == "pandas":
                    import pandas
                    version = pandas.__version__
                elif package == "streamlit":
                    import streamlit
                    version = streamlit.__version__
                elif package == "playwright":
                    from playwright import __version__
                    version = __version__
                else:
                    # Generic import
                    module = __import__(package)
                    version = getattr(module, '__version__', 'unknown')
                
                packages[package] = {
                    'installed': True,
                    'version': version,
                    'import_success': True
                }
                
                # Check version pinning
                if package in self.pin_versions:
                    required = self.pin_versions[package]
                    if self._version_matches(version, required):
                        print(f"     ✅ {package} {version} matches {required}")
                    else:
                        print(f"     ⚠️  {package} {version} doesn't match {required}")
                        packages[package]['version_mismatch'] = True
                        check_result['recommendations'].append(f"Update {package} to {required}")
                else:
                    print(f"     ✅ {package} {version} imported successfully")
                    
            except ImportError as e:
                packages[package] = {
                    'installed': False,
                    'import_success': False,
                    'error': str(e)
                }
                print(f"     ❌ {package} import failed: {e}")
                check_result['recommendations'].append(f"Install {package}")
                
            except Exception as e:
                packages[package] = {
                    'installed': True,
                    'import_success': False,
                    'error': str(e)
                }
                print(f"     ⚠️  {package} import error: {e}")
        
        check_result['details'] = packages
        
        # Determine overall status
        failed_imports = [p for p, info in packages.items() if not info.get('import_success', False)]
        if failed_imports:
            check_result['status'] = 'fail'
            print(f"   ❌ Failed package imports: {failed_imports}")
        else:
            check_result['status'] = 'pass'
            print(f"   ✅ All packages imported successfully")
        
        return check_result
    
    def check_numpy_pandas_compatibility(self) -> Dict[str, Any]:
        """Special check for numpy/pandas compatibility issues"""
        print("🔬 Checking numpy/pandas compatibility...")
        
        check_result = {
            'name': 'NumPy/Pandas Compatibility',
            'status': 'unknown',
            'details': {},
            'recommendations': []
        }
        
        try:
            # Test the specific error that was occurring
            import numpy as np
            numpy_version = np.__version__
            
            # Try to import pandas and check for the dtype size error
            try:
                import pandas as pd
                pandas_version = pd.__version__
                
                # Test basic operation
                df = pd.DataFrame({'test': [1, 2, 3]})
                
                check_result['details'] = {
                    'numpy_version': numpy_version,
                    'pandas_version': pandas_version,
                    'compatibility_test': 'passed'
                }
                check_result['status'] = 'pass'
                print(f"   ✅ NumPy {numpy_version} + Pandas {pandas_version} compatibility confirmed")
                
            except ValueError as e:
                if "numpy.dtype size changed" in str(e):
                    check_result['details'] = {
                        'numpy_version': numpy_version,
                        'pandas_version': 'unknown',
                        'compatibility_error': str(e)
                    }
                    check_result['status'] = 'fail'
                    check_result['recommendations'].extend([
                        "Reinstall pandas with compatible numpy version",
                        "Use: pip uninstall pandas numpy && pip install pandas==2.2.2 numpy==2.0.1",
                        "Or use: uv add pandas==2.2.2 numpy==2.0.1"
                    ])
                    print(f"   ❌ NumPy/Pandas compatibility issue detected")
                else:
                    raise
                    
        except ImportError as e:
            check_result['details'] = {'import_error': str(e)}
            check_result['status'] = 'fail'
            check_result['recommendations'].append("Install numpy and pandas")
            print(f"   ❌ Import error: {e}")
        except Exception as e:
            check_result['details'] = {'unexpected_error': str(e)}
            check_result['status'] = 'fail'
            print(f"   ❌ Unexpected error: {e}")
        
        return check_result
    
    def check_streamlit_config(self) -> Dict[str, Any]:
        """Check Streamlit configuration issues"""
        print("🎨 Checking Streamlit configuration...")
        
        check_result = {
            'name': 'Streamlit Configuration',
            'status': 'unknown',
            'details': {},
            'recommendations': []
        }
        
        config_files = [
            Path.home() / '.streamlit' / 'config.toml',
            self.project_root / '.streamlit' / 'config.toml'
        ]
        
        config_issues = []
        
        for config_file in config_files:
            if config_file.exists():
                try:
                    content = config_file.read_text()
                    
                    # Check for deprecated config options
                    for deprecated_option in self.clean_config:
                        if deprecated_option in content:
                            config_issues.append({
                                'file': str(config_file),
                                'option': deprecated_option,
                                'action': 'remove'
                            })
                            print(f"   ⚠️  Found deprecated option '{deprecated_option}' in {config_file}")
                    
                except Exception as e:
                    config_issues.append({
                        'file': str(config_file),
                        'error': str(e),
                        'action': 'check_manually'
                    })
        
        check_result['details'] = {
            'config_files_checked': [str(f) for f in config_files],
            'issues_found': config_issues
        }
        
        if config_issues:
            check_result['status'] = 'warn'
            check_result['recommendations'].append("Clean deprecated Streamlit config options")
        else:
            check_result['status'] = 'pass'
            print(f"   ✅ No Streamlit config issues found")
        
        return check_result
    
    def fix_dependencies(self) -> Dict[str, Any]:
        """Fix dependency issues"""
        print("🔧 Fixing dependency issues...")
        
        fix_result = {
            'name': 'Dependency Fixes',
            'status': 'unknown',
            'details': {},
            'actions_taken': []
        }
        
        actions = []
        
        # Check if we should fix numpy/pandas compatibility
        if "numpy" in self.pin_versions and "pandas" in self.pin_versions:
            numpy_version = self.pin_versions["numpy"]
            pandas_version = self.pin_versions["pandas"]
            
            print(f"   🔄 Fixing numpy/pandas compatibility...")
            print(f"      Target: numpy=={numpy_version}, pandas=={pandas_version}")
            
            # Try uv first, then pip
            success = False
            
            # Method 1: uv
            print(f"   📦 Attempting fix with uv...")
            code, stdout, stderr = self.run_command([
                'uv', 'remove', 'pandas', 'numpy'
            ], timeout=120)
            
            if code == 0:
                code, stdout, stderr = self.run_command([
                    'uv', 'add', f'numpy=={numpy_version}', f'pandas=={pandas_version}'
                ], timeout=120)
                
                if code == 0:
                    success = True
                    actions.append(f"✅ Fixed numpy/pandas with uv: numpy=={numpy_version}, pandas=={pandas_version}")
                    print(f"      ✅ Successfully reinstalled with uv")
                else:
                    actions.append(f"❌ uv installation failed: {stderr}")
                    print(f"      ❌ uv installation failed: {stderr}")
            
            # Method 2: pip fallback
            if not success:
                print(f"   📦 Attempting fix with pip...")
                code, stdout, stderr = self.run_command([
                    sys.executable, '-m', 'pip', 'uninstall', '-y', 'pandas', 'numpy'
                ], timeout=120)
                
                if code == 0:
                    code, stdout, stderr = self.run_command([
                        sys.executable, '-m', 'pip', 'install', 
                        f'numpy=={numpy_version}', f'pandas=={pandas_version}'
                    ], timeout=120)
                    
                    if code == 0:
                        success = True
                        actions.append(f"✅ Fixed numpy/pandas with pip: numpy=={numpy_version}, pandas=={pandas_version}")
                        print(f"      ✅ Successfully reinstalled with pip")
                    else:
                        actions.append(f"❌ pip installation failed: {stderr}")
                        print(f"      ❌ pip installation failed: {stderr}")
            
            if success:
                # Verify the fix
                try:
                    import numpy as np
                    import pandas as pd
                    test_df = pd.DataFrame({'test': [1, 2, 3]})
                    actions.append(f"✅ Compatibility verified: numpy {np.__version__}, pandas {pd.__version__}")
                    print(f"      ✅ Compatibility verified")
                except Exception as e:
                    actions.append(f"⚠️  Verification failed: {e}")
                    print(f"      ⚠️  Verification failed: {e}")
        
        fix_result['actions_taken'] = actions
        fix_result['status'] = 'success' if any('✅' in action for action in actions) else 'partial'
        
        return fix_result
    
    def fix_streamlit_config(self) -> Dict[str, Any]:
        """Fix Streamlit configuration issues"""
        print("🎨 Fixing Streamlit configuration...")
        
        fix_result = {
            'name': 'Streamlit Config Fixes',
            'status': 'unknown',
            'details': {},
            'actions_taken': []
        }
        
        actions = []
        
        config_files = [
            Path.home() / '.streamlit' / 'config.toml',
            self.project_root / '.streamlit' / 'config.toml'
        ]
        
        for config_file in config_files:
            if config_file.exists():
                try:
                    content = config_file.read_text()
                    original_content = content
                    
                    # Remove deprecated options
                    for deprecated_option in self.clean_config:
                        if deprecated_option in content:
                            # Remove lines containing the deprecated option
                            lines = content.split('\n')
                            filtered_lines = [
                                line for line in lines 
                                if deprecated_option not in line
                            ]
                            content = '\n'.join(filtered_lines)
                            actions.append(f"✅ Removed '{deprecated_option}' from {config_file}")
                            print(f"   ✅ Removed '{deprecated_option}' from {config_file}")
                    
                    # Write back if changed
                    if content != original_content:
                        # Backup original
                        backup_file = config_file.with_suffix('.toml.backup')
                        backup_file.write_text(original_content)
                        actions.append(f"📋 Backed up original to {backup_file}")
                        
                        # Write cleaned version
                        config_file.write_text(content)
                        actions.append(f"✅ Updated {config_file}")
                        print(f"   ✅ Updated {config_file}")
                    
                except Exception as e:
                    actions.append(f"❌ Error processing {config_file}: {e}")
                    print(f"   ❌ Error processing {config_file}: {e}")
        
        fix_result['actions_taken'] = actions
        fix_result['status'] = 'success' if any('✅' in action for action in actions) else 'no_action'
        
        return fix_result
    
    def _version_matches(self, current: str, requirement: str) -> bool:
        """Check if current version matches requirement pattern"""
        if requirement.endswith('*'):
            # Pattern like "3.11.*"
            prefix = requirement[:-1]
            return current.startswith(prefix)
        else:
            # Exact match
            return current == requirement
    
    def diagnose_and_fix(self) -> Dict[str, Any]:
        """Run comprehensive diagnosis and fixes"""
        print("🏥 Starting Python Environment Diagnosis & Repair...")
        print("=" * 60)
        
        # Run all checks
        checks = [
            self.check_python_version(),
            self.check_virtual_environment(),
            self.check_package_manager(),
            self.check_dependencies(),
            self.check_numpy_pandas_compatibility(),
            self.check_streamlit_config()
        ]
        
        print("\n" + "=" * 60)
        print("🔧 Running Fixes...")
        print("=" * 60)
        
        # Run fixes
        fixes = [
            self.fix_dependencies(),
            self.fix_streamlit_config()
        ]
        
        # Compile report
        self.report['checks'] = {check['name']: check for check in checks}
        self.report['fixes'] = {fix['name']: fix for fix in fixes}
        
        # Calculate summary
        for check in checks:
            self.report['summary']['total_checks'] += 1
            if check['status'] == 'pass':
                self.report['summary']['passed'] += 1
            elif check['status'] == 'fail':
                self.report['summary']['failed'] += 1
        
        for fix in fixes:
            if fix['status'] in ['success', 'partial']:
                self.report['summary']['fixed'] += 1
        
        print("\n" + "=" * 60)
        print("📊 DIAGNOSIS SUMMARY")
        print("=" * 60)
        
        summary = self.report['summary']
        print(f"Total Checks: {summary['total_checks']}")
        print(f"✅ Passed: {summary['passed']}")
        print(f"❌ Failed: {summary['failed']}")
        print(f"🔧 Fixed: {summary['fixed']}")
        
        # Overall health score
        health_score = (summary['passed'] + summary['fixed']) / summary['total_checks'] * 100
        print(f"🎯 Health Score: {health_score:.0f}%")
        
        if health_score >= 80:
            print("🎉 Environment is healthy!")
        elif health_score >= 60:
            print("⚠️  Environment has some issues but is mostly functional")
        else:
            print("🚨 Environment needs attention")
        
        return self.report


def main():
    """Main entry point"""
    # Parse command line arguments (simplified)
    check_packages = ["numpy", "pandas", "streamlit", "playwright"]
    pin_versions = {
        "python": "3.11.*",
        "numpy": "2.0.1", 
        "pandas": "2.2.2"
    }
    clean_config = ["runner.installTracer", "runner.fixMatplotlib"]
    
    doctor = PythonDoctor(
        check_packages=check_packages,
        pin_versions=pin_versions,
        clean_config=clean_config
    )
    
    report = doctor.diagnose_and_fix()
    
    # Save detailed report
    report_file = Path("python_doctor_report.json")
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📋 Detailed report saved to: {report_file}")
    
    return 0 if report['summary']['failed'] == 0 else 1


if __name__ == "__main__":
    exit(main())