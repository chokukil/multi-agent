"""
Data Processing Components for Cherry AI Universal Engine

Enhanced file processing and multi-dataset intelligence systems:
- enhanced_file_processor: Multi-format file processing with visual feedback
- llm_data_intelligence: LLM-powered dataset relationship discovery
- data_profiler: Automatic data profiling and quality assessment
"""

# Import data components with graceful degradation
try:
    from .enhanced_file_processor import EnhancedFileProcessor
except ImportError:
    EnhancedFileProcessor = None

# Export available components
__all__ = [
    name for name, obj in [
        ("EnhancedFileProcessor", EnhancedFileProcessor),
    ] if obj is not None
]