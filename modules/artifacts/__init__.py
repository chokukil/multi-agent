"""
Artifact Rendering Components for Cherry AI Universal Engine

Interactive artifact rendering system with smart download capabilities:
- interactive_plotly_renderer: Enhanced Plotly charts with full interactivity
- virtual_scroll_table_renderer: High-performance table rendering with virtual scrolling
- syntax_highlight_code_renderer: Code rendering with syntax highlighting and copy functionality
- responsive_image_renderer: Image rendering with click-to-enlarge and zoom controls
- smart_download_manager: Two-tier download system (raw + enhanced formats)
"""

# Import artifact components with graceful degradation
try:
    from .interactive_plotly_renderer import InteractivePlotlyRenderer
except ImportError:
    InteractivePlotlyRenderer = None

try:
    from .virtual_scroll_table_renderer import VirtualScrollTableRenderer
except ImportError:
    VirtualScrollTableRenderer = None

try:
    from .syntax_highlight_code_renderer import SyntaxHighlightCodeRenderer
except ImportError:
    SyntaxHighlightCodeRenderer = None

try:
    from .responsive_image_renderer import ResponsiveImageRenderer
except ImportError:
    ResponsiveImageRenderer = None

try:
    from .smart_download_manager import SmartDownloadManager
except ImportError:
    SmartDownloadManager = None

# Export available components
__all__ = [
    name for name, obj in [
        ("InteractivePlotlyRenderer", InteractivePlotlyRenderer),
        ("VirtualScrollTableRenderer", VirtualScrollTableRenderer),
        ("SyntaxHighlightCodeRenderer", SyntaxHighlightCodeRenderer),
        ("ResponsiveImageRenderer", ResponsiveImageRenderer),
        ("SmartDownloadManager", SmartDownloadManager)
    ] if obj is not None
]