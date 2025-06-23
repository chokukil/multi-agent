import pytest
from unittest.mock import MagicMock, patch, mock_open

# Mock Streamlit before importing the module under test
st_mock = MagicMock()
# We need to mock the context manager part of st.expander
st_mock.expander.return_value.__enter__.return_value = st_mock
st_mock.container.return_value.__enter__.return_value = st_mock


modules = {
    "streamlit": st_mock,
    "streamlit.components.v1": st_mock.components.v1,
    "core.data_manager": MagicMock(),
    "plotly.io": MagicMock()
}

with patch.dict("sys.modules", modules):
    from ui.artifact_manager import render_artifact

@pytest.fixture(autouse=True)
def reset_mocks():
    """Reset mocks before each test."""
    st_mock.reset_mock()

def test_render_artifact_with_html_string():
    """
    Tests if render_artifact correctly calls st.components.v1.html for 'html' type.
    """
    html_content = "<h1>Test</h1>"
    render_artifact("html", html_content, st_mock.container())
    st_mock.components.v1.html.assert_called_once_with(html_content, height=600, scrolling=True)

def test_render_artifact_with_html_filepath():
    """
    Tests if render_artifact correctly reads an HTML file and calls st.components.v1.html.
    """
    html_content = "<h1>From File</h1>"
    file_path = "dummy.html"
    
    with patch("builtins.open", mock_open(read_data=html_content)) as mocked_file:
        render_artifact("file_path", file_path, st_mock.container())
        
        mocked_file.assert_called_once_with(file_path, 'r', encoding='utf-8')
        st_mock.components.v1.html.assert_called_once_with(html_content, height=600, scrolling=True)

def test_render_artifact_with_non_html_filepath():
    """
    Tests if render_artifact provides a download button for non-HTML files.
    """
    file_path = "data.csv"
    
    with patch("builtins.open", mock_open(read_data="a,b,c")) as mocked_file:
        render_artifact("file_path", file_path, st_mock.container())
        
        # Should not try to render as HTML
        st_mock.components.v1.html.assert_not_called()
        
        # Should try to create a download button
        mocked_file.assert_called_once_with(file_path, "rb")
        st_mock.download_button.assert_called_once()
        
def test_render_artifact_with_text():
    """
    Tests if render_artifact correctly calls st.text for 'text' type.
    """
    text_content = "This is a simple text."
    render_artifact("text", text_content, st_mock.container())
    st_mock.text.assert_called_once_with(text_content)

def test_render_artifact_with_markdown():
    """
    Tests if render_artifact correctly calls st.markdown for 'markdown' type.
    """
    markdown_content = "## Markdown Header"
    render_artifact("markdown", markdown_content, st_mock.container())
    st_mock.markdown.assert_called_once_with(markdown_content) 