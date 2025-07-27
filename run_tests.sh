#!/bin/bash
# Cherry AI Streamlit Platform - Test Runner Script

set -e

echo "🧪 Cherry AI Streamlit Platform - Test Suite"
echo "============================================="

# Configuration
TEST_TYPE=${1:-all}
COVERAGE_THRESHOLD=${2:-80}

echo "Test type: $TEST_TYPE"
echo "Coverage threshold: $COVERAGE_THRESHOLD%"

# Check if pytest is installed
if ! python -c "import pytest" 2>/dev/null; then
    echo "❌ pytest is not installed. Installing..."
    pip install pytest pytest-asyncio pytest-cov
fi

# Create test directories
mkdir -p tests/reports
mkdir -p htmlcov

# Run different test types
case $TEST_TYPE in
    "unit")
        echo "🔬 Running unit tests..."
        python -m pytest tests/ -m "unit" \
            --cov=modules \
            --cov-report=html:htmlcov \
            --cov-report=term-missing \
            --cov-report=xml:tests/reports/coverage.xml \
            --cov-fail-under=$COVERAGE_THRESHOLD \
            --junit-xml=tests/reports/junit.xml \
            -v
        ;;
    
    "integration")
        echo "🔗 Running integration tests..."
        python -m pytest tests/ -m "integration" \
            --cov=modules \
            --cov-report=html:htmlcov \
            --cov-report=term-missing \
            --cov-report=xml:tests/reports/coverage.xml \
            --junit-xml=tests/reports/junit.xml \
            -v
        ;;
    
    "performance")
        echo "⚡ Running performance tests..."
        python -m pytest tests/ -m "performance" \
            --cov=modules \
            --cov-report=html:htmlcov \
            --cov-report=term-missing \
            --junit-xml=tests/reports/junit.xml \
            -v
        ;;
    
    "security")
        echo "🔒 Running security tests..."
        python -m pytest tests/ -m "security" \
            --cov=modules \
            --cov-report=html:htmlcov \
            --cov-report=term-missing \
            --junit-xml=tests/reports/junit.xml \
            -v
        ;;
    
    "all")
        echo "🚀 Running all tests..."
        python -m pytest tests/ \
            --cov=modules \
            --cov-report=html:htmlcov \
            --cov-report=term-missing \
            --cov-report=xml:tests/reports/coverage.xml \
            --cov-fail-under=$COVERAGE_THRESHOLD \
            --junit-xml=tests/reports/junit.xml \
            -v
        ;;
    
    "quick")
        echo "⚡ Running quick tests (excluding slow tests)..."
        python -m pytest tests/ -m "not slow" \
            --cov=modules \
            --cov-report=term-missing \
            --cov-fail-under=$COVERAGE_THRESHOLD \
            -v
        ;;
    
    *)
        echo "❌ Unknown test type: $TEST_TYPE"
        echo "Available types: unit, integration, performance, security, all, quick"
        exit 1
        ;;
esac

# Check test results
TEST_EXIT_CODE=$?

if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✅ All tests passed!"
    echo ""
    echo "📊 Test Reports:"
    echo "   Coverage Report: htmlcov/index.html"
    echo "   JUnit Report: tests/reports/junit.xml"
    echo "   Coverage XML: tests/reports/coverage.xml"
    
    # Show coverage summary
    if [ -f "tests/reports/coverage.xml" ]; then
        echo ""
        echo "📈 Coverage Summary:"
        python -c "
import xml.etree.ElementTree as ET
try:
    tree = ET.parse('tests/reports/coverage.xml')
    root = tree.getroot()
    coverage = root.attrib.get('line-rate', '0')
    print(f'   Overall Coverage: {float(coverage)*100:.1f}%')
except:
    print('   Coverage data not available')
"
    fi
    
else
    echo ""
    echo "❌ Some tests failed!"
    echo "Check the output above for details."
fi

# Performance benchmarking (if performance tests were run)
if [ "$TEST_TYPE" = "performance" ] || [ "$TEST_TYPE" = "all" ]; then
    echo ""
    echo "⚡ Performance Benchmark Results:"
    echo "   (Results would be displayed here in a full implementation)"
fi

# Security scan results (if security tests were run)
if [ "$TEST_TYPE" = "security" ] || [ "$TEST_TYPE" = "all" ]; then
    echo ""
    echo "🔒 Security Scan Results:"
    echo "   (Security scan results would be displayed here)"
fi

echo ""
echo "🎯 Test Summary:"
echo "   Test Type: $TEST_TYPE"
echo "   Coverage Threshold: $COVERAGE_THRESHOLD%"
echo "   Exit Code: $TEST_EXIT_CODE"

exit $TEST_EXIT_CODE