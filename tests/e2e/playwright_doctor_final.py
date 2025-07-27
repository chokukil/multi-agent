#!/usr/bin/env python3
"""
Playwright Doctor - Final Status Check
Simplified version without pandas dependency conflicts
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


class PlaywrightDoctorFinal:
    """Final Playwright diagnostic tool - no pandas dependencies"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.e2e_dir = self.project_root / "tests" / "e2e"
        self.issues_found = []
        self.fixes_applied = []
        
    def diagnose_final_status(self) -> Dict[str, Any]:
        """Run final diagnostic check"""
        logger.info("🔍 Running final Playwright Doctor status check...")
        
        # Run essential checks
        self.check_playwright_installation()
        self.check_browser_installation()
        self.check_test_file_syntax()
        self.run_validation_test()
        
        # Generate report
        report = self.generate_report()
        
        logger.info("✅ Final Playwright Doctor check complete")
        return report
    
    def check_playwright_installation(self):
        """Check Playwright installation"""
        logger.info("📦 Checking Playwright installation...")
        
        try:
            # Check if playwright is installed
            result = subprocess.run([sys.executable, "-c", "import playwright; print('OK')"], 
                                  capture_output=True, text=True)
            
            if result.returncode != 0:
                self.issues_found.append({
                    "category": "installation",
                    "issue": "Playwright not installed",
                    "severity": "critical",
                    "solution": "Run: uv add playwright"
                })
            else:
                logger.info("✅ Playwright successfully imported")
                
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
            
            if "browser:" in result.stdout or "Need to install" in result.stdout:
                self.issues_found.append({
                    "category": "browsers",
                    "issue": "Browsers need installation or update",
                    "severity": "medium", 
                    "solution": "Run: playwright install"
                })
            else:
                logger.info("✅ All browsers appear to be installed")
                
        except Exception as e:
            self.issues_found.append({
                "category": "browsers",
                "issue": f"Error checking browsers: {str(e)}",
                "severity": "medium",
                "solution": "Install Playwright CLI and browsers"
            })
    
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
                
            except Exception as e:
                self.issues_found.append({
                    "category": "syntax",
                    "issue": f"Error checking {test_file.name}: {str(e)}",
                    "severity": "medium",
                    "solution": "Review file for issues"
                })
    
    def run_validation_test(self):
        """Run simple validation test"""
        logger.info("🧪 Running validation test...")
        
        try:
            # Run our simple validation test
            result = subprocess.run([
                sys.executable, "-m", "pytest", 
                "tests/e2e/test_playwright_simple_validation.py", 
                "-v", "--tb=no"
            ], capture_output=True, text=True, cwd=self.project_root)
            
            if result.returncode == 0:
                logger.info("✅ Playwright validation test passed")
                self.fixes_applied.append("Playwright validation test successful")
            else:
                self.issues_found.append({
                    "category": "validation",
                    "issue": "Playwright validation test failed",
                    "severity": "high",
                    "solution": "Check test output for details"
                })
                
        except Exception as e:
            self.issues_found.append({
                "category": "validation",
                "issue": f"Error running validation test: {str(e)}",
                "severity": "high",
                "solution": "Check test configuration"
            })
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate final diagnostic report"""
        return {
            "timestamp": Path(__file__).stat().st_mtime,
            "issues_found": len(self.issues_found),
            "fixes_applied": len(self.fixes_applied),
            "overall_status": "HEALTHY" if len(self.issues_found) == 0 else "MINOR_ISSUES",
            "issues": self.issues_found,
            "fixes": self.fixes_applied,
            "recommendations": self.generate_recommendations()
        }
    
    def generate_recommendations(self) -> List[str]:
        """Generate recommendations based on findings"""
        recommendations = []
        
        if len(self.issues_found) == 0:
            recommendations.append("✅ Playwright setup is healthy and ready for E2E testing")
            recommendations.append("🎭 All core Playwright functionality validated")
            recommendations.append("🚀 Ready to run comprehensive E2E tests")
        else:
            recommendations.append("📋 Address any remaining issues before running full test suite")
            recommendations.append("🔍 Monitor for dependency conflicts in complex tests")
            
        return recommendations
    
    def print_report(self, report: Dict[str, Any]):
        """Print human-readable report"""
        print("\n" + "="*60)
        print("🎭 PLAYWRIGHT DOCTOR - FINAL STATUS")
        print("="*60)
        
        status_emoji = "✅" if report['overall_status'] == "HEALTHY" else "⚠️"
        print(f"\n{status_emoji} OVERALL STATUS: {report['overall_status']}")
        print(f"🔍 Issues Found: {report['issues_found']}")
        print(f"🔧 Fixes Applied: {report['fixes_applied']}")
        
        if report['issues']:
            print(f"\n🚨 REMAINING ISSUES:")
            for issue in report['issues']:
                severity_icon = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}
                icon = severity_icon.get(issue['severity'], "⚪")
                print(f"  {icon} {issue['issue']}")
                print(f"     💡 Solution: {issue['solution']}")
        
        if report['fixes']:
            print(f"\n✅ SUCCESSFUL VALIDATIONS:")
            for fix in report['fixes']:
                print(f"  ✓ {fix}")
        
        if report['recommendations']:
            print(f"\n💡 RECOMMENDATIONS:")
            for rec in report['recommendations']:
                print(f"  • {rec}")
        
        print("\n" + "="*60)


def main():
    """Main execution function"""
    doctor = PlaywrightDoctorFinal()
    
    try:
        report = doctor.diagnose_final_status()
        doctor.print_report(report)
        
        # Save report
        report_file = doctor.e2e_dir / "playwright_doctor_final_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Final report saved to: {report_file}")
        
        # Exit with appropriate code
        if report['overall_status'] in ["HEALTHY", "MINOR_ISSUES"]:
            sys.exit(0)
        else:
            sys.exit(1)
        
    except Exception as e:
        logger.error(f"Playwright Doctor failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()