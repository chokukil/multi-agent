# Cherry AI Streamlit Platform

**Enhanced ChatGPT/Claude-style Data Science Platform with Multi-Agent Collaboration**

Built on proven patterns from the LLM First Universal Engine, this Streamlit platform provides an intuitive, powerful interface for data analysis with real-time agent collaboration visualization.

## 🌟 Key Features

### Enhanced User Interface
- **ChatGPT/Claude-style Interface**: Familiar chat-based interaction with enhanced visual feedback
- **Real-time Agent Collaboration**: Live visualization of multi-agent work with progress bars and status indicators
- **Progressive Disclosure**: Summary-first results with expandable detailed views
- **One-Click Execution**: Intelligent analysis recommendations with immediate execution buttons
- **Responsive Design**: Mobile-first approach with touch-friendly controls

### Advanced File Processing
- **Multi-Format Support**: CSV, Excel (.xlsx, .xls), JSON, Parquet, and PKL formats
- **Visual Data Cards**: Interactive dataset previews with quality indicators and relationship discovery
- **Drag-and-Drop Upload**: Enhanced file upload with visual boundaries and progress tracking
- **Automatic Profiling**: Comprehensive data quality assessment and metadata extraction

### Multi-Agent Integration
- **Proven A2A SDK 0.2.9 Patterns**: Leverages validated Universal Engine agent communication
- **10 Specialized Agents**: Data cleaning, visualization, ML, EDA, and more
- **Sequential/Parallel Execution**: Intelligent workflow orchestration based on dependencies
- **Error Recovery**: Progressive retry and circuit breaker patterns for resilience

### Interactive Artifacts
- **Enhanced Plotly Rendering**: Fully interactive charts with hover, zoom, and pan capabilities
- **Virtual Scroll Tables**: High-performance data tables with sorting and filtering
- **Syntax-Highlighted Code**: Copy-to-clipboard functionality with language-specific formatting
- **Smart Download System**: Raw artifacts always available plus context-aware enhanced formats

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- A2A SDK 0.2.9
- A2A agents running on ports 8306-8315 (for full functionality)

### Installation & Launch

1. **Install Dependencies**:
   ```bash
   pip install -r requirements_streamlit.txt
   ```

2. **Start A2A Agents** (Required for multi-agent features):
   ```bash
   # Start your A2A agents on ports 8306-8315
   # Example agent startup commands (adjust per your setup):
   # a2a_agent --port 8306 --type data_cleaning
   # a2a_agent --port 8307 --type data_loader
   # ... (for ports 8308-8315)
   ```

3. **Launch Cherry AI Streamlit Platform**:
   ```bash
   ./start_cherry_ai_streamlit.sh
   ```
   
   Or manually:
   ```bash
   streamlit run cherry_ai_streamlit_app.py --server.port 8501
   ```

4. **Access Platform**:
   Open your browser to `http://localhost:8501`

## 📁 Architecture

### Modular Structure
```
modules/
├── core/                    # Universal Engine Integration
│   ├── universal_orchestrator.py
│   ├── llm_recommendation_engine.py
│   └── streaming_controller.py
├── ui/                      # Enhanced Interface Components
│   ├── enhanced_chat_interface.py
│   ├── file_upload.py
│   ├── layout_manager.py
│   └── status_display.py
├── data/                    # Intelligent File Processing
│   ├── enhanced_file_processor.py
│   ├── llm_data_intelligence.py
│   └── data_profiler.py
├── artifacts/               # Interactive Rendering System
│   ├── plotly_renderer.py
│   ├── table_renderer.py
│   ├── code_renderer.py
│   └── smart_download_manager.py
├── a2a/                     # A2A SDK Integration
│   ├── agent_client.py
│   ├── agent_discovery.py
│   └── workflow_orchestrator.py
└── utils/                   # Supporting Utilities
    ├── llm_error_handler.py
    ├── performance_monitor.py
    └── security_validator.py
```

### Component Integration
- **Main App**: `cherry_ai_streamlit_app.py` - Single-page application under 300 lines
- **Layout Manager**: Responsive design with mobile-first approach
- **Chat Interface**: Enhanced with typing indicators and agent collaboration panels
- **File Processing**: Multi-format support with real-time progress tracking
- **Universal Engine**: LLM-powered orchestration and meta-reasoning

## 💡 Usage Examples

### 1. Data Upload and Analysis
1. **Upload Files**: Drag and drop CSV, Excel, or JSON files
2. **Review Data Cards**: Interactive previews with quality indicators
3. **Get Suggestions**: Automatic analysis recommendations with one-click execution
4. **Chat Interaction**: Natural language queries about your data

### 2. Multi-Dataset Analysis
1. **Upload Multiple Files**: Platform discovers relationships automatically
2. **Visual Relationship Diagrams**: See potential joins and connections
3. **Comparative Analysis**: Cross-dataset insights and correlations
4. **Unified Results**: Integrated findings from multiple agents

### 3. Advanced Workflows
1. **Statistical Analysis**: Comprehensive summaries and distributions
2. **Data Visualization**: Interactive Plotly charts with export options
3. **Machine Learning**: Model training with performance metrics
4. **Quality Assessment**: Data cleaning recommendations and validation

## 🎨 UI/UX Features

### ChatGPT/Claude-Style Interface
- **Message Bubbles**: User messages right-aligned, AI responses left-aligned with Cherry AI avatar
- **Typing Indicators**: Animated dots with agent-specific progress information
- **Auto-Scroll**: Smooth scrolling to latest messages with session persistence
- **Keyboard Shortcuts**: Shift+Enter for line breaks, Enter to send

### Real-Time Agent Collaboration
- **Progress Visualization**: Individual agent progress bars (0-100%)
- **Status Indicators**: Working (⚡), completed (✅), failed (❌) states
- **Agent Avatars**: Visual representation of each specialized agent
- **Data Flow**: Inter-agent communication and data transfer visualization

### Progressive Disclosure
- **Summary First**: 3-5 key insights with visual highlights
- **Expandable Details**: "📄 View All Details" button reveals complete analysis
- **Agent History**: Step-by-step breakdown of agent work
- **Download Options**: Raw artifacts + context-aware enhanced formats

## 🔧 Configuration

### Environment Variables
```bash
# Streamlit Configuration
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Universal Engine Configuration
LLM_PROVIDER=OLLAMA
OLLAMA_MODEL=okamototk/gemma3-tools:4b
OLLAMA_BASE_URL=http://localhost:11434

# A2A Agent Configuration
A2A_AGENTS_BASE_URL=http://localhost
```

### Agent Mapping
- **8306**: Data Cleaning Agent - LLM-based intelligent data cleaning
- **8307**: Data Loader Agent - Multi-format file loading with encoding support
- **8308**: Data Visualization Agent - Interactive Plotly chart generation
- **8309**: Data Wrangling Agent - Data transformation and manipulation
- **8310**: Feature Engineering Agent - Feature creation and selection
- **8311**: SQL Database Agent - Database queries and operations
- **8312**: EDA Tools Agent - Exploratory data analysis and pattern discovery
- **8313**: H2O ML Agent - Machine learning and AutoML capabilities
- **8314**: MLflow Tools Agent - Model management and experiment tracking
- **8315**: Pandas Analyst Agent - Core pandas operations and analysis

## 📊 Performance & Scalability

### Performance Targets
- **File Processing**: 10MB files in under 10 seconds
- **Memory Usage**: Under 1GB per session
- **Concurrent Users**: Support for 50+ simultaneous users
- **Response Time**: Analysis completion under 30 seconds for typical datasets

### Optimization Features
- **Virtual Scrolling**: Efficient handling of large datasets in tables
- **Lazy Loading**: On-demand data loading for better performance
- **Caching**: Smart caching of processed datasets and analysis results
- **Progressive Loading**: Incremental content delivery for better UX

## 🛡️ Security & Privacy

### Data Protection
- **Session Isolation**: Complete separation between user sessions
- **Temporary Files**: Secure handling and automatic cleanup
- **No Data Persistence**: Files are processed in memory when possible
- **Access Control**: Proper authentication and authorization mechanisms

### Security Validation
- **File Type Validation**: Strict checking of uploaded file formats
- **Malicious Code Scanning**: Security checks for potentially harmful content
- **Input Sanitization**: Protection against XSS and injection attacks
- **Error Handling**: Secure error messages without information leakage

## 🔍 Troubleshooting

### Common Issues

1. **Port Conflicts**
   - Ensure port 8501 is available for Streamlit
   - Check that no other services are using port 8501

2. **A2A Connection Issues**
   - Verify agents are running on ports 8306-8315
   - Check network connectivity and firewall settings
   - Platform will work in basic mode without agents

3. **File Upload Errors**
   - Ensure file formats are supported (CSV, Excel, JSON, Parquet, PKL)
   - Check file size limits and encoding issues
   - Verify file permissions and disk space

4. **Memory Issues**
   - Use virtual scrolling for large datasets (>10MB)
   - Monitor memory usage with system tools
   - Consider data sampling for very large files

5. **Missing Dependencies**
   - Run `pip install -r requirements_streamlit.txt`
   - Check Python version compatibility (3.8+)
   - Verify all modules are properly installed

### Debug Mode
```bash
# Enable debug logging
export STREAMLIT_LOGGER_LEVEL=debug
streamlit run cherry_ai_streamlit_app.py --logger.level=debug
```

## 🤝 Contributing

This platform is built on proven Universal Engine patterns. When contributing:

1. **Follow Existing Patterns**: Leverage validated components from core/universal_engine/
2. **Maintain Modularity**: Keep components in appropriate modules/ subdirectories
3. **Test Thoroughly**: Ensure all features work with and without Universal Engine
4. **Document Changes**: Update this README and add inline documentation

## 📄 License

This project is part of the Cherry AI ecosystem and follows the same licensing terms as the main Cherry AI project.

---

**Cherry AI Streamlit Platform** - Where proven LLM patterns meet intuitive data science interfaces 🍒✨