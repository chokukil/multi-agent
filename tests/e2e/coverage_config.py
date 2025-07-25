"""
Test Coverage Configuration and Reporting

Implements comprehensive test coverage measurement for the Cherry AI platform:
- Code coverage analysis
- E2E test coverage tracking
- Quality metrics reporting
- Coverage thresholds enforcement
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class CoverageMetrics:
    """Coverage metrics data structure"""
    line_coverage: float = 0.0
    branch_coverage: float = 0.0
    function_coverage: float = 0.0
    statement_coverage: float = 0.0
    total_lines: int = 0
    covered_lines: int = 0
    missed_lines: int = 0
    total_branches: int = 0
    covered_branches: int = 0
    total_functions: int = 0
    covered_functions: int = 0


@dataclass
class CoverageReport:
    """Comprehensive coverage report"""
    timestamp: datetime = field(default_factory=datetime.now)
    overall_metrics: CoverageMetrics = field(default_factory=CoverageMetrics)
    module_metrics: Dict[str, CoverageMetrics] = field(default_factory=dict)
    test_execution_summary: Dict[str, Any] = field(default_factory=dict)
    quality_gates_passed: bool = False
    coverage_threshold: float = 0.9
    recommendations: List[str] = field(default_factory=list)


class CoverageManager:
    """
    Test coverage management and reporting system
    """
    
    def __init__(self, project_root: str = None, coverage_threshold: float = 0.9):
        self.project_root = Path(project_root or os.getcwd())
        self.coverage_threshold = coverage_threshold
        self.coverage_data_file = self.project_root / ".coverage"
        self.coverage_report_dir = self.project_root / "tests" / "e2e" / "coverage_reports"
        self.coverage_report_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Coverage Manager initialized with threshold {coverage_threshold}")
    
    def setup_coverage_environment(self) -> bool:
        """Set up coverage measurement environment"""
        try:
            # Install coverage.py if not available
            try:
                import coverage
            except ImportError:
                logger.info("Installing coverage.py...")
                subprocess.run([sys.executable, "-m", "pip", "install", "coverage"], check=True)
                import coverage
            
            # Create coverage configuration
            self._create_coverage_config()
            
            logger.info("Coverage environment setup complete")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup coverage environment: {e}")
            return False
    
    def _create_coverage_config(self):
        """Create .coveragerc configuration file"""
        coverage_config = """[run]
branch = True
source = modules/, cherry_ai_streamlit_app.py
omit = 
    tests/*
    */__pycache__/*
    */test_*
    setup.py
    conftest.py
    */migrations/*
    */venv/*
    */.venv/*

[report]
precision = 2
show_missing = True
skip_covered = False

[html]
directory = tests/e2e/coverage_reports/html

[xml]
output = tests/e2e/coverage_reports/coverage.xml

[json]
output = tests/e2e/coverage_reports/coverage.json
"""
        
        config_file = self.project_root / ".coveragerc"
        config_file.write_text(coverage_config)
        logger.info(f"Created coverage config: {config_file}")
    
    async def run_coverage_analysis(self, test_command: str = None) -> CoverageReport:
        """Run comprehensive coverage analysis"""
        try:
            # Default test command if none provided
            if test_command is None:
                test_command = "python -m pytest tests/e2e/ -v --tb=short"
            
            # Run tests with coverage
            coverage_command = f"coverage run --rcfile=.coveragerc -m pytest tests/e2e/ -v"
            
            logger.info(f"Running coverage analysis: {coverage_command}")
            
            # Execute coverage run
            result = subprocess.run(
                coverage_command.split(),
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            # Generate coverage reports
            await self._generate_coverage_reports()
            
            # Parse and analyze results
            report = await self._parse_coverage_results()
            
            # Check quality gates
            report.quality_gates_passed = self._check_quality_gates(report)
            
            # Generate recommendations
            report.recommendations = self._generate_recommendations(report)
            
            # Save report
            await self._save_coverage_report(report)
            
            logger.info(f"Coverage analysis complete. Overall coverage: {report.overall_metrics.line_coverage:.1f}%")
            return report
            
        except Exception as e:
            logger.error(f"Coverage analysis failed: {e}")
            # Return empty report with error info
            report = CoverageReport()
            report.recommendations = [f"Coverage analysis failed: {str(e)}"]
            return report
    
    async def _generate_coverage_reports(self):
        """Generate various coverage report formats"""
        commands = [
            "coverage report",
            "coverage html",
            "coverage xml", 
            "coverage json"
        ]
        
        for cmd in commands:
            try:
                subprocess.run(
                    cmd.split(),
                    cwd=self.project_root,
                    check=True,
                    capture_output=True
                )
                logger.debug(f"Generated coverage report: {cmd}")
            except subprocess.CalledProcessError as e:
                logger.warning(f"Failed to generate {cmd}: {e}")
    
    async def _parse_coverage_results(self) -> CoverageReport:
        """Parse coverage results from generated reports"""
        report = CoverageReport()
        report.coverage_threshold = self.coverage_threshold
        
        try:
            # Parse JSON coverage report
            json_report_path = self.coverage_report_dir / "coverage.json"
            if json_report_path.exists():
                with open(json_report_path, 'r') as f:
                    coverage_data = json.load(f)
                
                # Extract overall metrics
                totals = coverage_data.get('totals', {})
                report.overall_metrics = CoverageMetrics(
                    line_coverage=totals.get('percent_covered', 0.0),
                    statement_coverage=totals.get('percent_covered_display', 0.0),
                    total_lines=totals.get('num_statements', 0),
                    covered_lines=totals.get('covered_lines', 0),
                    missed_lines=totals.get('missing_lines', 0),
                    total_branches=totals.get('num_branches', 0),
                    covered_branches=totals.get('covered_branches', 0)
                )
                
                # Extract module-level metrics
                files = coverage_data.get('files', {})
                for file_path, file_data in files.items():
                    module_name = self._get_module_name(file_path)
                    summary = file_data.get('summary', {})
                    
                    report.module_metrics[module_name] = CoverageMetrics(
                        line_coverage=summary.get('percent_covered', 0.0),
                        total_lines=summary.get('num_statements', 0),
                        covered_lines=summary.get('covered_lines', 0),
                        missed_lines=summary.get('missing_lines', 0),
                        total_branches=summary.get('num_branches', 0),
                        covered_branches=summary.get('covered_branches', 0)
                    )
                
                logger.info("Successfully parsed JSON coverage report")
            
        except Exception as e:
            logger.error(f"Failed to parse coverage results: {e}")
        
        return report
    
    def _get_module_name(self, file_path: str) -> str:
        """Extract module name from file path"""
        path = Path(file_path)
        if "modules/" in file_path:
            # Extract module path relative to modules/
            parts = path.parts
            try:
                modules_index = parts.index("modules")
                module_parts = parts[modules_index + 1:]
                return "/".join(module_parts).replace(".py", "")
            except ValueError:
                pass
        
        # Fallback to filename
        return path.stem
    
    def _check_quality_gates(self, report: CoverageReport) -> bool:
        """Check if coverage meets quality gate thresholds"""
        overall_coverage = report.overall_metrics.line_coverage / 100.0
        
        # Primary gate: overall coverage threshold
        if overall_coverage < self.coverage_threshold:
            logger.warning(f"Coverage {overall_coverage:.1%} below threshold {self.coverage_threshold:.1%}")
            return False
        
        # Secondary gates: critical modules must have high coverage
        critical_modules = [
            "core/security_validation_system",
            "core/enhanced_exception_handling", 
            "core/agent_circuit_breaker",
            "core/enhanced_file_upload"
        ]
        
        for module in critical_modules:
            if module in report.module_metrics:
                module_coverage = report.module_metrics[module].line_coverage / 100.0
                if module_coverage < 0.8:  # 80% threshold for critical modules
                    logger.warning(f"Critical module {module} coverage {module_coverage:.1%} below 80%")
                    return False
        
        logger.info("All quality gates passed")
        return True
    
    def _generate_recommendations(self, report: CoverageReport) -> List[str]:
        """Generate coverage improvement recommendations"""
        recommendations = []
        
        overall_coverage = report.overall_metrics.line_coverage
        
        if overall_coverage < self.coverage_threshold * 100:
            recommendations.append(
                f"🎯 Increase overall test coverage from {overall_coverage:.1f}% to {self.coverage_threshold * 100:.1f}%"
            )
        
        # Identify modules with low coverage
        low_coverage_modules = []
        for module, metrics in report.module_metrics.items():
            if metrics.line_coverage < 70:  # Less than 70% coverage
                low_coverage_modules.append((module, metrics.line_coverage))
        
        if low_coverage_modules:
            low_coverage_modules.sort(key=lambda x: x[1])  # Sort by coverage ascending
            recommendations.append(
                f"📊 Focus testing on low-coverage modules: {', '.join([f'{mod} ({cov:.1f}%)' for mod, cov in low_coverage_modules[:3]])}"
            )
        
        # Branch coverage recommendations
        if report.overall_metrics.total_branches > 0:
            branch_coverage = (report.overall_metrics.covered_branches / report.overall_metrics.total_branches) * 100
            if branch_coverage < 80:
                recommendations.append(
                    f"🌿 Improve branch coverage from {branch_coverage:.1f}% by testing error paths and edge cases"
                )
        
        # Specific recommendations based on missed lines
        if report.overall_metrics.missed_lines > 0:
            recommendations.append(
                f"📝 Add tests for {report.overall_metrics.missed_lines} uncovered lines across {len(report.module_metrics)} modules"
            )
        
        # E2E specific recommendations
        recommendations.extend([
            "🔧 Add integration tests for agent communication paths",
            "🛡️ Ensure security validation edge cases are tested",
            "⚡ Test error recovery and circuit breaker scenarios",
            "📁 Validate file upload performance under various conditions"
        ])
        
        return recommendations
    
    async def _save_coverage_report(self, report: CoverageReport):
        """Save coverage report to file"""
        report_file = self.coverage_report_dir / f"coverage_report_{report.timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        
        # Convert report to JSON-serializable format
        report_data = {
            "timestamp": report.timestamp.isoformat(),
            "overall_metrics": {
                "line_coverage": report.overall_metrics.line_coverage,
                "branch_coverage": report.overall_metrics.branch_coverage,
                "total_lines": report.overall_metrics.total_lines,
                "covered_lines": report.overall_metrics.covered_lines,
                "missed_lines": report.overall_metrics.missed_lines,
                "total_branches": report.overall_metrics.total_branches,
                "covered_branches": report.overall_metrics.covered_branches
            },
            "module_metrics": {
                module: {
                    "line_coverage": metrics.line_coverage,
                    "total_lines": metrics.total_lines,
                    "covered_lines": metrics.covered_lines,
                    "missed_lines": metrics.missed_lines
                }
                for module, metrics in report.module_metrics.items()
            },
            "quality_gates_passed": report.quality_gates_passed,
            "coverage_threshold": report.coverage_threshold,
            "recommendations": report.recommendations
        }
        
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        logger.info(f"Coverage report saved: {report_file}")
    
    def generate_coverage_badge(self, coverage_percentage: float) -> str:
        """Generate coverage badge URL"""
        if coverage_percentage >= 90:
            color = "brightgreen"
        elif coverage_percentage >= 80:
            color = "green"
        elif coverage_percentage >= 70:
            color = "yellow"
        elif coverage_percentage >= 60:
            color = "orange"
        else:
            color = "red"
        
        return f"https://img.shields.io/badge/coverage-{coverage_percentage:.1f}%25-{color}"
    
    def get_coverage_summary(self) -> Dict[str, Any]:
        """Get latest coverage summary"""
        try:
            # Find latest coverage report
            report_files = list(self.coverage_report_dir.glob("coverage_report_*.json"))
            if not report_files:
                return {"error": "No coverage reports found"}
            
            latest_report = max(report_files, key=lambda f: f.stat().st_mtime)
            
            with open(latest_report, 'r') as f:
                data = json.load(f)
            
            return {
                "timestamp": data["timestamp"],
                "line_coverage": data["overall_metrics"]["line_coverage"],
                "quality_gates_passed": data["quality_gates_passed"],
                "total_modules": len(data["module_metrics"]),
                "recommendations_count": len(data["recommendations"]),
                "badge_url": self.generate_coverage_badge(data["overall_metrics"]["line_coverage"])
            }
            
        except Exception as e:
            logger.error(f"Failed to get coverage summary: {e}")
            return {"error": str(e)}


# Global coverage manager instance
_global_coverage_manager: Optional[CoverageManager] = None


def get_coverage_manager(project_root: str = None, threshold: float = 0.9) -> CoverageManager:
    """Get global coverage manager instance"""
    global _global_coverage_manager
    if _global_coverage_manager is None:
        _global_coverage_manager = CoverageManager(project_root, threshold)
    return _global_coverage_manager


async def run_comprehensive_coverage_analysis(threshold: float = 0.9) -> CoverageReport:
    """Run comprehensive coverage analysis with quality gates"""
    manager = get_coverage_manager(threshold=threshold)
    
    # Setup environment
    if not manager.setup_coverage_environment():
        raise RuntimeError("Failed to setup coverage environment")
    
    # Run analysis
    report = await manager.run_coverage_analysis()
    
    return report