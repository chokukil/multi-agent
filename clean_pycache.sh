#!/bin/bash

echo "Cleaning __pycache__ directories..."

# Delete all __pycache__ directories recursively
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null

# Also clean .pyc files that might be outside __pycache__
find . -type f -name "*.pyc" -delete 2>/dev/null

# Clean .pyo files as well
find . -type f -name "*.pyo" -delete 2>/dev/null

# Count remaining __pycache__ directories to verify cleanup
remaining=$(find . -type d -name "__pycache__" 2>/dev/null | wc -l)

echo ""
if [ "$remaining" -eq 0 ]; then
    echo "✅ Cleanup completed successfully! All __pycache__ directories removed."
else
    echo "⚠️  Warning: $remaining __pycache__ directories still remain."
fi

echo "Done!" 