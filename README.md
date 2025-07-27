# 🍒 Cherry AI Streamlit Platform

## Enhanced ChatGPT/Claude-style Data Analysis Platform

A comprehensive, production-ready data analysis platform built with Streamlit, featuring LLM orchestration, A2A SDK integration, and advanced UI/UX optimizations.

### 🌟 Key Features

- **Enhanced ChatGPT/Claude Interface**: Natural conversation with AI for data analysis
- **Universal Engine Integration**: Proven patterns from 100% implemented Universal Engine
- **A2A SDK 0.2.9**: Multi-agent collaboration with 10 specialized agents (ports 8306-8315)
- **LLM Orchestration**: 4-stage meta-reasoning for intelligent analysis
- **Advanced Security**: LLM-powered threat detection and validation
- **Performance Optimization**: Caching, memory management, and concurrent processing
- **Comprehensive UX**: Visual feedback, workflow guidance, and accessibility features
- **Production Ready**: Docker deployment, monitoring, and scaling support

### 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Cherry AI Platform                       │
├─────────────────────────────────────────────────────────────┤
│  UI Layer: Enhanced Chat Interface + UX Optimization       │
├─────────────────────────────────────────────────────────────┤
│  Core: Universal Orchestrator + LLM Recommendation Engine  │
├─────────────────────────────────────────────────────────────┤
│  A2A Agents: 10 Specialized Agents (8306-8315)            │
├─────────────────────────────────────────────────────────────┤
│  Utils: Performance + Security + Error Handling + Caching  │
├─────────────────────────────────────────────────────────────┤
│  Infrastructure: Redis + PostgreSQL + Ollama + MLflow      │
└─────────────────────────────────────────────────────────────┘
```

### 🚀 Quick Start

#### Option 1: Docker Deployment (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd cherry-ai-streamlit-platform

# Deploy with Docker Compose
./deploy.sh production 2

# Access the application
open http://localhost:8501
```

#### Option 2: Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env with your configuration

# Run the application
streamlit run cherry_ai_integrated_app.py
```

### 📊 A2A Agents

The platform includes 10 specialized A2A agents:

| Port | Agent | Description |
|------|-------|-------------|
| 8306 | 🧹 Data Cleaning | LLM-based intelligent data cleaning |
| 8307 | 📁 Data Loader | Multi-format file loading and processing |
| 8308 | 📊 Visualization | Interactive Plotly-based visualizations |
| 8309 | 🔧 Data Wrangling | Data transformation and manipulation |
| 8310 | ⚙️ Feature Engineering | ML feature creation and optimization |
| 8311 | 🗄️ SQL Database | Database operations and queries |
| 8312 | 🔍 EDA Tools | Exploratory data analysis |
| 8313 | 🤖 H2O ML | Machine learning and AutoML |
| 8314 | 📈 MLflow Tools | Model management and tracking |
| 8315 | 🐼 Pandas Hub | Advanced pandas operations |

### 🔧 Configuration

#### Environment Variables

Key configuration options in `.env`:

```bash
# LLM Configuration
LLM_PROVIDER=OLLAMA
OLLAMA_MODEL=llama3
OPENAI_API_KEY=your_key_here

# Performance Settings
MAX_WORKERS=20
MAX_CONCURRENT_USERS=50
CACHE_ENABLED=true

# Security Settings
MAX_FILE_SIZE_MB=200
ALLOWED_FILE_TYPES=csv,xlsx,json,parquet

# Database
POSTGRES_URL=postgresql://user:pass@host:5432/db
REDIS_URL=redis://host:6379
```

#### LLM Providers

Supported LLM providers:
- **Ollama** (default): Local LLM deployment
- **OpenAI**: GPT-3.5/GPT-4 integration
- **Anthropic**: Claude integration

### 🛡️ Security Features

- **File Security Validation**: LLM-powered threat detection
- **Session Management**: Secure session isolation
- **Rate Limiting**: Protection against abuse
- **Access Control**: Role-based permissions
- **Data Privacy**: Sensitive data detection and handling

### ⚡ Performance Features

- **Multi-level Caching**: Memory + persistent caching
- **Memory Management**: Lazy loading and garbage collection optimization
- **Concurrent Processing**: Multi-threaded agent execution
- **Performance Monitoring**: Real-time metrics and optimization
- **Load Balancing**: Horizontal scaling support

### 🎨 UX Features

- **Visual Feedback**: Immediate response to all user actions
- **Workflow Guidance**: Step-by-step process guidance
- **Progressive Disclosure**: Summary-first with expandable details
- **Accessibility**: Screen reader support and keyboard navigation
- **Responsive Design**: Mobile-first approach

### 📈 Monitoring

Access monitoring dashboards:
- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090
- **MLflow**: http://localhost:5000

### 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
./run_tests.sh all

# Run specific test types
./run_tests.sh unit
./run_tests.sh integration
./run_tests.sh performance
./run_tests.sh security

# Quick tests (excluding slow tests)
./run_tests.sh quick
```

Test coverage target: 80%+

### 🔄 Development Workflow

1. **File Upload**: Drag & drop with security validation
2. **Data Preview**: Automatic insights and quality indicators
3. **Analysis Selection**: AI-powered recommendations
4. **Processing**: Real-time agent collaboration visualization
5. **Results Review**: Interactive artifacts with progressive disclosure
6. **Download**: Smart download system with multiple formats

### 📦 Deployment Options

#### Production Deployment

```bash
# Full production stack
./deploy.sh production 3

# With custom scaling
docker-compose up -d --scale cherry-ai-app=5 cherry-ai-app
```

#### Development Deployment

```bash
# Development mode
./deploy.sh development

# Local testing
streamlit run cherry_ai_integrated_app.py
```

### 🔧 Scaling

The platform supports horizontal scaling:

- **Application Scaling**: Multiple Streamlit instances behind load balancer
- **Agent Scaling**: Independent scaling of A2A agents
- **Database Scaling**: PostgreSQL with read replicas
- **Cache Scaling**: Redis cluster support

### 📚 API Documentation

#### Core Components

- **UniversalOrchestrator**: Main orchestration engine
- **EnhancedFileProcessor**: File processing and validation
- **LLMRecommendationEngine**: AI-powered analysis suggestions
- **SecurityValidator**: File and session security
- **PerformanceMonitor**: System performance tracking

#### Agent Communication

All agents follow A2A SDK 0.2.9 protocol:
- JSON-RPC 2.0 communication
- Health check endpoints
- Capability discovery
- Error handling and recovery

### 🐛 Troubleshooting

#### Common Issues

1. **Agent Connection Failed**
   ```bash
   # Check agent health
   curl http://localhost:8306/health
   
   # Restart specific agent
   docker-compose restart data-cleaning-agent
   ```

2. **High Memory Usage**
   ```bash
   # Check memory stats
   docker stats
   
   # Adjust memory limits in docker-compose.yml
   ```

3. **Slow Performance**
   ```bash
   # Check performance metrics
   curl http://localhost:8501/metrics
   
   # Scale application
   docker-compose up -d --scale cherry-ai-app=3 cherry-ai-app
   ```

### 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `./run_tests.sh all`
5. Submit a pull request

### 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

### 🙏 Acknowledgments

- Built on proven Universal Engine patterns (100% implementation success)
- Leverages A2A SDK 0.2.9 for agent communication
- Inspired by ChatGPT/Claude user experience design
- Uses Streamlit for rapid web application development

### 📞 Support

For support and questions:
- Create an issue in the repository
- Check the troubleshooting section
- Review the comprehensive test suite for examples

---

**Cherry AI Streamlit Platform** - Intelligent Data Analysis with LLM Orchestration 🍒