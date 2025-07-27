#!/usr/bin/env python3
"""
Playwright Doctor - Comprehensive diagnostic and fix tool for Playwright issues

This script identifies and fixes common Playwright issues in the Cherry AI project:
- Browser installation problems
- Configuration issues
- Test file syntax errors
- Dependency conflicts
- Performance issues
"""

import subprocess
import sys
import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PlaywrightDoctor:
    """Comprehensive Playwright diagnostic and repair tool"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.e2e_dir = self.project_root / "tests" / "e2e"
        self.issues_found = []
        self.fixes_applied = []
        
    def diagnose_and_fix(self) -> Dict[str, Any]:
        """Run complete diagnostic and fix process"""
        logger.info("🔍 Starting Playwright Doctor diagnosis...")
        
        # Run all diagnostic checks
        self.check_playwright_installation()
        self.check_browser_installation()
        self.check_test_configuration()
        self.check_test_file_syntax()
        self.check_dependencies()
        self.check_performance_settings()
        
        # Generate report
        report = self.generate_report()
        
        logger.info("✅ Playwright Doctor diagnosis complete")
        return report
    
    def check_playwright_installation(self):
        """Check Playwright installation"""
        logger.info("📦 Checking Playwright installation...")
        
        try:
            # Check if playwright is installed
            result = subprocess.run([sys.executable, "-c", "import playwright; print(playwright.__version__)"], 
                                  capture_output=True, text=True)
            
            if result.returncode != 0:
                self.issues_found.append({
                    "category": "installation",
                    "issue": "Playwright not installed",
                    "severity": "critical",
                    "solution": "pip install playwright"
                })
                self.fix_playwright_installation()
            else:
                version = result.stdout.strip()
                logger.info(f"✅ Playwright version {version} installed")
                
                # Check for outdated version
                if self.is_version_outdated(version, "1.40.0"):
                    self.issues_found.append({
                        "category": "installation", 
                        "issue": f"Playwright version {version} is outdated",
                        "severity": "medium",
                        "solution": "pip install --upgrade playwright"
                    })
                    self.fix_playwright_version()
                    
        except Exception as e:
            self.issues_found.append({
                "category": "installation",
                "issue": f"Error checking Playwright installation: {str(e)}",
                "severity": "critical",
                "solution": "Reinstall Playwright"
            })
    
    def check_browser_installation(self):
        """Check browser installation"""
        logger.info("🌐 Checking browser installation...")
        
        try:
            # Check if browsers are installed
            result = subprocess.run(["playwright", "install", "--dry-run"], 
                                  capture_output=True, text=True)
            
            if "browser:" in result.stdout:
                self.issues_found.append({
                    "category": "browsers",
                    "issue": "Browsers need installation or update",
                    "severity": "high", 
                    "solution": "Run playwright install"
                })
                self.fix_browser_installation()
            else:
                logger.info("✅ All browsers up to date")
                
        except Exception as e:
            self.issues_found.append({
                "category": "browsers",
                "issue": f"Error checking browsers: {str(e)}",
                "severity": "high",
                "solution": "Install Playwright CLI and browsers"
            })
    
    def check_test_configuration(self):
        """Check test configuration files"""
        logger.info("⚙️ Checking test configuration...")
        
        # Check pytest.ini
        pytest_ini = self.e2e_dir / "pytest.ini"
        if not pytest_ini.exists():
            self.issues_found.append({
                "category": "configuration",
                "issue": "pytest.ini missing",
                "severity": "medium",
                "solution": "Create pytest.ini configuration"
            })
            self.fix_pytest_config()
        else:
            logger.info("✅ pytest.ini found")
        
        # Check conftest.py
        conftest = self.e2e_dir / "conftest.py"
        if not conftest.exists():
            self.issues_found.append({
                "category": "configuration",
                "issue": "conftest.py missing",
                "severity": "medium",
                "solution": "Create conftest.py with fixtures"
            })
        else:
            logger.info("✅ conftest.py found")
        
        # Check requirements
        requirements = self.e2e_dir / "requirements.txt"
        if not requirements.exists():
            self.issues_found.append({
                "category": "configuration",
                "issue": "requirements.txt missing",
                "severity": "medium",
                "solution": "Create requirements.txt"
            })
        else:
            logger.info("✅ requirements.txt found")
    
    def check_test_file_syntax(self):
        """Check test file syntax"""
        logger.info("📝 Checking test file syntax...")
        
        test_files = list(self.e2e_dir.glob("test_*.py"))
        
        for test_file in test_files:
            try:
                # Check syntax by compiling
                with open(test_file, 'r') as f:
                    content = f.read()
                
                compile(content, str(test_file), 'exec')
                logger.info(f"✅ {test_file.name} syntax OK")
                
            except SyntaxError as e:
                self.issues_found.append({
                    "category": "syntax",
                    "issue": f"Syntax error in {test_file.name}: {str(e)}",
                    "severity": "critical",
                    "solution": f"Fix syntax error at line {e.lineno}"
                })
                self.fix_syntax_error(test_file, e)
                
            except Exception as e:
                self.issues_found.append({
                    "category": "syntax",
                    "issue": f"Error checking {test_file.name}: {str(e)}",
                    "severity": "medium",
                    "solution": "Review file for issues"
                })
    
    def check_dependencies(self):
        """Check dependency conflicts"""
        logger.info("📚 Checking dependencies...")
        
        try:
            # Check for pandas conflict
            result = subprocess.run([sys.executable, "-c", "import pandas; print(pandas.__version__)"],
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                pandas_version = result.stdout.strip()
                if pandas_version.startswith("2."):
                    self.issues_found.append({
                        "category": "dependencies",
                        "issue": f"Pandas version {pandas_version} conflicts with pandasai",
                        "severity": "medium",
                        "solution": "Downgrade pandas to 1.5.x"
                    })
                    self.fix_pandas_conflict()
                else:
                    logger.info(f"✅ Pandas version {pandas_version} compatible")
                    
        except Exception as e:
            logger.warning(f"Could not check pandas version: {e}")
    
    def check_performance_settings(self):
        """Check performance-related settings"""
        logger.info("⚡ Checking performance settings...")
        
        # Check for timeout settings in pytest.ini
        pytest_ini = self.e2e_dir / "pytest.ini"
        if pytest_ini.exists():
            content = pytest_ini.read_text()
            if "timeout" not in content:
                self.issues_found.append({
                    "category": "performance",
                    "issue": "No timeout configuration in pytest.ini",
                    "severity": "low",
                    "solution": "Add timeout = 300 to pytest.ini"
                })
        
        # Check for excessive sleep() calls
        sleep_count = 0
        test_files = list(self.e2e_dir.glob("test_*.py"))
        
        for test_file in test_files:
            try:
                content = test_file.read_text()
                sleep_matches = re.findall(r'asyncio\.sleep\s*\(', content)
                sleep_count += len(sleep_matches)
            except Exception:
                pass
        
        if sleep_count > 5:
            self.issues_found.append({
                "category": "performance",
                "issue": f"Found {sleep_count} sleep() calls in tests",
                "severity": "medium",
                "solution": "Replace sleep() with proper wait conditions"
            })
    
    # Fix methods
    
    def fix_playwright_installation(self):
        """Fix Playwright installation"""
        logger.info("🔧 Fixing Playwright installation...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "playwright"], check=True)
            self.fixes_applied.append("Installed Playwright")
        except Exception as e:
            logger.error(f"Failed to install Playwright: {e}")
    
    def fix_playwright_version(self):
        """Fix Playwright version"""
        logger.info("🔧 Upgrading Playwright...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "playwright"], check=True)
            self.fixes_applied.append("Upgraded Playwright")
        except Exception as e:
            logger.error(f"Failed to upgrade Playwright: {e}")
    
    def fix_browser_installation(self):
        """Fix browser installation"""
        logger.info("🔧 Installing/updating browsers...")
        try:
            subprocess.run(["playwright", "install", "chromium"], check=True)
            self.fixes_applied.append("Installed/updated Chromium browser")
        except Exception as e:
            logger.error(f"Failed to install browsers: {e}")
    
    def fix_pytest_config(self):
        """Fix pytest configuration"""
        logger.info("🔧 Creating pytest.ini...")
        pytest_config = """[tool:pytest]
testpaths = tests/e2e
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    -v
    --tb=short
    --strict-markers
    --strict-config
    --disable-warnings
markers =
    ui: UI-specific tests
    agent: Agent collaboration tests
    performance: Performance tests
    security: Security tests
    slow: Slow running tests (>30s)
    integration: Integration tests
    smoke: Quick smoke tests
filterwarnings =
    ignore::DeprecationWarning
    ignore::PendingDeprecationWarning
asyncio_mode = auto
timeout = 300
log_cli = true
log_cli_level = INFO
"""
        try:
            (self.e2e_dir / "pytest.ini").write_text(pytest_config)
            self.fixes_applied.append("Created pytest.ini configuration")
        except Exception as e:
            logger.error(f"Failed to create pytest.ini: {e}")
    
    def fix_syntax_error(self, file_path: Path, error: SyntaxError):
        """Fix common syntax errors"""
        logger.info(f"🔧 Attempting to fix syntax error in {file_path.name}...")
        
        try:
            content = file_path.read_text()
            
            # Fix common assertion message quote issues
            if "assert" in str(error):
                # Fix nested quote issues in assertions
                fixed_content = re.sub(
                    r'assert\s+([^,]+),\s*"([^"]*)"([^"]*)"([^"]*)"',
                    r'assert \1, "\2\3\4"',
                    content
                )
                
                if fixed_content != content:
                    file_path.write_text(fixed_content)
                    self.fixes_applied.append(f"Fixed assertion syntax in {file_path.name}")
                    
        except Exception as e:
            logger.error(f"Failed to fix syntax error: {e}")
    
    def fix_pandas_conflict(self):
        """Fix pandas version conflict"""
        logger.info("🔧 Fixing pandas version conflict...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "pandas>=1.5.3,<2.0.0"], check=True)
            self.fixes_applied.append("Downgraded pandas to compatible version")
        except Exception as e:
            logger.error(f"Failed to fix pandas conflict: {e}")
    
    # Utility methods
    
    def is_version_outdated(self, current: str, minimum: str) -> bool:
        """Check if version is outdated"""
        try:
            current_parts = [int(x) for x in current.split('.')]
            minimum_parts = [int(x) for x in minimum.split('.')]
            
            for i in range(max(len(current_parts), len(minimum_parts))):
                c = current_parts[i] if i < len(current_parts) else 0
                m = minimum_parts[i] if i < len(minimum_parts) else 0
                
                if c < m:
                    return True
                elif c > m:
                    return False
            
            return False
        except Exception:
            return False
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate diagnostic report"""
        return {
            "timestamp": Path(__file__).stat().st_mtime,
            "issues_found": len(self.issues_found),
            "fixes_applied": len(self.fixes_applied),
            "overall_status": "HEALTHY" if len(self.issues_found) == 0 else "ISSUES_FOUND",
            "issues": self.issues_found,
            "fixes": self.fixes_applied,
            "recommendations": self.generate_recommendations()
        }
    
    def generate_recommendations(self) -> List[str]:
        """Generate recommendations based on findings"""
        recommendations = []
        
        if any(issue["category"] == "performance" for issue in self.issues_found):
            recommendations.append("Consider replacing sleep() calls with proper wait conditions")
            
        if any(issue["category"] == "syntax" for issue in self.issues_found):
            recommendations.append("Run regular syntax checks before committing test files")
            
        if any(issue["category"] == "dependencies" for issue in self.issues_found):
            recommendations.append("Use virtual environments to avoid dependency conflicts")
            
        if len(self.issues_found) == 0:
            recommendations.append("Playwright setup is healthy - run regular checks")
            
        return recommendations
    
    def print_report(self, report: Dict[str, Any]):
        """Print human-readable report"""
        print("\n" + "="*60)
        print("🎭 PLAYWRIGHT DOCTOR REPORT")
        print("="*60)
        
        print(f"\n📊 OVERALL STATUS: {report['overall_status']}")
        print(f"🔍 Issues Found: {report['issues_found']}")
        print(f"🔧 Fixes Applied: {report['fixes_applied']}")
        
        if report['issues']:
            print(f"\n🚨 ISSUES FOUND:")
            for issue in report['issues']:
                severity_icon = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}
                icon = severity_icon.get(issue['severity'], "⚪")
                print(f"  {icon} {issue['issue']}")
                print(f"     💡 Solution: {issue['solution']}")
        
        if report['fixes']:
            print(f"\n✅ FIXES APPLIED:")
            for fix in report['fixes']:
                print(f"  ✓ {fix}")
        
        if report['recommendations']:
            print(f"\n💡 RECOMMENDATIONS:")
            for rec in report['recommendations']:
                print(f"  • {rec}")
        
        print("\n" + "="*60)


def main():
    """Main execution function"""
    doctor = PlaywrightDoctor()
    
    try:
        report = doctor.diagnose_and_fix()
        doctor.print_report(report)
        
        # Save report
        report_file = doctor.e2e_dir / "playwright_doctor_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Detailed report saved to: {report_file}")
        
        # Exit with appropriate code
        if report['overall_status'] == "ISSUES_FOUND":
            if any(issue['severity'] in ['critical', 'high'] for issue in report['issues']):
                sys.exit(1)
        
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Playwright Doctor failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()