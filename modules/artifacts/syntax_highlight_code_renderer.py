"""
Syntax Highlight Code Renderer - Enhanced Features with Raw Code Download

Code rendering with:
- Syntax highlighting for multiple languages
- Line numbers
- Copy to clipboard functionality
- Code folding for long files
- Raw code file download always available
"""

import streamlit as st
import streamlit.components.v1 as components
import logging
import ast
from typing import Dict, Any, Optional
import pygments
from pygments.lexers import get_lexer_by_name, guess_lexer
from pygments.formatters import HtmlFormatter
from pygments.styles import get_style_by_name

logger = logging.getLogger(__name__)


class SyntaxHighlightCodeRenderer:
    """Syntax highlighted code renderer with advanced features"""
    
    def __init__(self):
        """Initialize code renderer"""
        self.supported_languages = {
            'python': 'Python',
            'javascript': 'JavaScript',
            'typescript': 'TypeScript',
            'java': 'Java',
            'cpp': 'C++',
            'c': 'C',
            'csharp': 'C#',
            'go': 'Go',
            'rust': 'Rust',
            'sql': 'SQL',
            'r': 'R',
            'julia': 'Julia',
            'bash': 'Bash',
            'yaml': 'YAML',
            'json': 'JSON',
            'xml': 'XML',
            'html': 'HTML',
            'css': 'CSS',
            'markdown': 'Markdown'
        }
        
        self.themes = {
            'monokai': 'Monokai',
            'github': 'GitHub',
            'vs': 'Visual Studio',
            'dracula': 'Dracula',
            'solarized-dark': 'Solarized Dark',
            'solarized-light': 'Solarized Light'
        }
        
    def render_code(self,
                   code: str,
                   language: Optional[str] = None,
                   title: Optional[str] = None,
                   theme: str = 'monokai',
                   show_line_numbers: bool = True,
                   enable_copy: bool = True,
                   enable_download: bool = True,
                   max_height: int = 600) -> Dict[str, Any]:
        """
        Render syntax highlighted code
        
        Returns:
            Dict with 'raw_code' for download
        """
        try:
            if title:
                st.markdown(f"### 💻 {title}")
            
            # Language selection if not specified
            if not language:
                detected_language = self._detect_language(code)
                col1, col2 = st.columns([3, 1])
                with col1:
                    language = st.selectbox(
                        "프로그래밍 언어",
                        options=list(self.supported_languages.keys()),
                        index=list(self.supported_languages.keys()).index(detected_language) if detected_language in self.supported_languages else 0,
                        format_func=lambda x: self.supported_languages[x],
                        key=f"lang_{hash(code[:50])}"
                    )
                with col2:
                    theme = st.selectbox(
                        "테마",
                        options=list(self.themes.keys()),
                        format_func=lambda x: self.themes[x],
                        key=f"theme_{hash(code[:50])}"
                    )
            
            # Code statistics
            lines = code.split('\n')
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("총 줄 수", f"{len(lines):,}")
            with col2:
                st.metric("문자 수", f"{len(code):,}")
            with col3:
                st.metric("파일 크기", f"{len(code.encode('utf-8')) / 1024:.2f} KB")
            
            # Syntax highlighting
            highlighted_code = self._highlight_code(code, language, theme, show_line_numbers)
            
            # Render with custom CSS
            code_html = f"""
            <style>
                .code-container {{
                    position: relative;
                    max-height: {max_height}px;
                    overflow-y: auto;
                    border: 1px solid #444;
                    border-radius: 8px;
                    margin: 10px 0;
                }}
                .copy-button {{
                    position: absolute;
                    top: 10px;
                    right: 10px;
                    padding: 5px 10px;
                    background: #667eea;
                    color: white;
                    border: none;
                    border-radius: 4px;
                    cursor: pointer;
                    font-size: 12px;
                    z-index: 10;
                }}
                .copy-button:hover {{
                    background: #764ba2;
                }}
                .highlight {{
                    margin: 0 !important;
                    padding: 15px !important;
                }}
                .highlight pre {{
                    margin: 0 !important;
                    padding-right: 50px !important;
                }}
            </style>
            <div class="code-container">
                {highlighted_code}
            </div>
            """
            
            # Add copy button JavaScript
            if enable_copy:
                code_html += f"""
                <script>
                function copyCode() {{
                    const code = {repr(code)};
                    navigator.clipboard.writeText(code).then(() => {{
                        alert('코드가 클립보드에 복사되었습니다!');
                    }});
                }}
                </script>
                <button class="copy-button" onclick="copyCode()">📋 복사</button>
                """
            
            # Render HTML
            components.html(code_html, height=min(max_height + 50, 800))
            
            # Download button
            if enable_download:
                file_extension = self._get_file_extension(language)
                st.download_button(
                    label=f"⬇️ 코드 다운로드 (.{file_extension})",
                    data=code.encode('utf-8'),
                    file_name=f"{title or 'code'}.{file_extension}",
                    mime="text/plain",
                    key=f"download_{hash(code[:50])}"
                )
            
            # Code analysis (for Python)
            if language == 'python':
                self._analyze_python_code(code)
            
            return {
                'raw_code': code,
                'language': language,
                'line_count': len(lines)
            }
            
        except Exception as e:
            logger.error(f"Error rendering code: {str(e)}")
            # Fallback to simple code display
            st.code(code, language=language)
            return {'raw_code': code}
    
    def _detect_language(self, code: str) -> str:
        """Detect programming language from code"""
        try:
            lexer = guess_lexer(code)
            # Map lexer name to our supported languages
            lexer_name = lexer.name.lower()
            
            language_map = {
                'python': 'python',
                'javascript': 'javascript',
                'java': 'java',
                'c++': 'cpp',
                'c': 'c',
                'sql': 'sql',
                'bash': 'bash',
                'json': 'json',
                'yaml': 'yaml',
                'html': 'html',
                'css': 'css'
            }
            
            for key, value in language_map.items():
                if key in lexer_name:
                    return value
            
            return 'python'  # Default
            
        except Exception:
            return 'python'  # Default fallback
    
    def _highlight_code(self, code: str, language: str, theme: str, show_line_numbers: bool) -> str:
        """Apply syntax highlighting to code"""
        try:
            # Get lexer and style
            lexer = get_lexer_by_name(language, stripall=True)
            style = get_style_by_name(theme)
            
            # Create formatter
            formatter = HtmlFormatter(
                style=style,
                linenos='table' if show_line_numbers else False,
                cssclass='highlight',
                wrapcode=True,
                prestyles='overflow-x:auto;'
            )
            
            # Highlight code
            highlighted = pygments.highlight(code, lexer, formatter)
            
            # Add CSS
            css = formatter.get_style_defs('.highlight')
            
            return f"<style>{css}</style>{highlighted}"
            
        except Exception as e:
            logger.error(f"Error highlighting code: {str(e)}")
            # Fallback to pre-formatted text
            return f"<pre><code>{code}</code></pre>"
    
    def _get_file_extension(self, language: str) -> str:
        """Get file extension for language"""
        extensions = {
            'python': 'py',
            'javascript': 'js',
            'typescript': 'ts',
            'java': 'java',
            'cpp': 'cpp',
            'c': 'c',
            'csharp': 'cs',
            'go': 'go',
            'rust': 'rs',
            'sql': 'sql',
            'r': 'r',
            'julia': 'jl',
            'bash': 'sh',
            'yaml': 'yml',
            'json': 'json',
            'xml': 'xml',
            'html': 'html',
            'css': 'css',
            'markdown': 'md'
        }
        return extensions.get(language, 'txt')
    
    def _analyze_python_code(self, code: str) -> None:
        """Analyze Python code and show metrics"""
        try:
            import ast
            
            # Parse code
            tree = ast.parse(code)
            
            # Count different node types
            functions = sum(1 for node in ast.walk(tree) if isinstance(node, ast.FunctionDef))
            classes = sum(1 for node in ast.walk(tree) if isinstance(node, ast.ClassDef))
            imports = sum(1 for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom)))
            
            # Display metrics
            with st.expander("📊 코드 분석", expanded=False):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("함수", functions)
                with col2:
                    st.metric("클래스", classes)
                with col3:
                    st.metric("Import", imports)
                
                # Complexity analysis
                complexity = self._calculate_complexity(tree)
                if complexity > 0:
                    st.metric("순환 복잡도", complexity)
                    if complexity > 10:
                        st.warning("⚠️ 높은 복잡도: 리팩토링을 고려하세요.")
                        
        except Exception as e:
            logger.debug(f"Could not analyze Python code: {str(e)}")
    
    def _calculate_complexity(self, tree: ast.AST) -> int:
        """Calculate cyclomatic complexity"""
        complexity = 1  # Base complexity
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For)):
                complexity += 1
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1
            elif isinstance(node, ast.With):
                complexity += 1
        
        return complexity
    
    def render_diff(self,
                   old_code: str,
                   new_code: str,
                   title: Optional[str] = None,
                   language: Optional[str] = None) -> None:
        """Render code diff with syntax highlighting"""
        try:
            import difflib
            
            if title:
                st.markdown(f"### 🔄 {title}")
            
            # Create diff
            old_lines = old_code.splitlines(keepends=True)
            new_lines = new_code.splitlines(keepends=True)
            
            diff = difflib.unified_diff(
                old_lines,
                new_lines,
                fromfile='이전',
                tofile='이후',
                lineterm=''
            )
            
            # Format diff with colors
            diff_html = []
            for line in diff:
                if line.startswith('+'):
                    diff_html.append(f'<span style="color: #28a745;">{line}</span>')
                elif line.startswith('-'):
                    diff_html.append(f'<span style="color: #dc3545;">{line}</span>')
                elif line.startswith('@'):
                    diff_html.append(f'<span style="color: #007bff; font-weight: bold;">{line}</span>')
                else:
                    diff_html.append(line)
            
            # Display diff
            diff_text = '\n'.join(diff_html)
            components.html(
                f"""
                <pre style="background: #1e1e1e; color: #d4d4d4; padding: 15px; 
                           border-radius: 8px; overflow-x: auto; font-family: monospace;">
                    {diff_text}
                </pre>
                """,
                height=400
            )
            
            # Statistics
            added = sum(1 for line in diff_html if line.startswith('<span style="color: #28a745;">'))
            removed = sum(1 for line in diff_html if line.startswith('<span style="color: #dc3545;">'))
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("추가된 줄", f"+{added}", delta_color="normal")
            with col2:
                st.metric("삭제된 줄", f"-{removed}", delta_color="inverse")
                
        except Exception as e:
            logger.error(f"Error rendering diff: {str(e)}")
            st.error(f"Diff 렌더링 오류: {str(e)}")