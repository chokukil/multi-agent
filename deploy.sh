#!/bin/bash
# Cherry AI Streamlit Platform - Deployment Script

set -e

echo "🍒 Cherry AI Streamlit Platform Deployment Script"
echo "=================================================="

# Configuration
ENVIRONMENT=${1:-production}
SCALE_REPLICAS=${2:-1}

echo "Environment: $ENVIRONMENT"
echo "Scale replicas: $SCALE_REPLICAS"

# Check if Docker and Docker Compose are installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data logs cache nginx/ssl monitoring/grafana/{dashboards,datasources}

# Copy environment file
if [ ! -f .env ]; then
    echo "📋 Creating .env file from template..."
    cp .env.example .env
    echo "⚠️  Please edit .env file with your configuration before continuing."
    echo "Press Enter to continue after editing .env file..."
    read
fi

# Build and start services
echo "🔨 Building and starting services..."

if [ "$ENVIRONMENT" = "development" ]; then
    echo "🚀 Starting development environment..."
    docker-compose -f docker-compose.yml up --build -d
else
    echo "🚀 Starting production environment..."
    
    # Pull latest images
    docker-compose pull
    
    # Build application
    docker-compose build cherry-ai-app
    
    # Start infrastructure services first
    echo "📊 Starting infrastructure services..."
    docker-compose up -d redis postgres ollama mlflow prometheus grafana
    
    # Wait for services to be ready
    echo "⏳ Waiting for services to be ready..."
    sleep 30
    
    # Start A2A agents
    echo "🤖 Starting A2A agents..."
    docker-compose up -d \
        data-cleaning-agent \
        data-loader-agent \
        visualization-agent \
        wrangling-agent \
        feature-engineering-agent \
        sql-database-agent \
        eda-tools-agent \
        h2o-ml-agent \
        mlflow-tools-agent \
        pandas-hub-agent
    
    # Wait for agents to be ready
    echo "⏳ Waiting for agents to be ready..."
    sleep 20
    
    # Start main application with scaling
    echo "🍒 Starting Cherry AI application..."
    docker-compose up -d --scale cherry-ai-app=$SCALE_REPLICAS cherry-ai-app
    
    # Start load balancer
    echo "⚖️ Starting load balancer..."
    docker-compose up -d nginx
fi

# Health check
echo "🏥 Performing health checks..."
sleep 10

# Check main application
if curl -f http://localhost:8501/_stcore/health > /dev/null 2>&1; then
    echo "✅ Main application is healthy"
else
    echo "❌ Main application health check failed"
fi

# Check agents
echo "🤖 Checking A2A agents..."
for port in {8306..8315}; do
    if curl -f http://localhost:$port/health > /dev/null 2>&1; then
        echo "✅ Agent on port $port is healthy"
    else
        echo "⚠️  Agent on port $port is not responding"
    fi
done

# Show service status
echo ""
echo "📊 Service Status:"
docker-compose ps

echo ""
echo "🎉 Deployment completed!"
echo ""
echo "🌐 Access URLs:"
echo "   Main Application: http://localhost:8501"
echo "   Grafana Dashboard: http://localhost:3000 (admin/admin)"
echo "   Prometheus: http://localhost:9090"
echo "   MLflow: http://localhost:5000"
echo ""
echo "📋 Useful commands:"
echo "   View logs: docker-compose logs -f cherry-ai-app"
echo "   Scale app: docker-compose up -d --scale cherry-ai-app=3 cherry-ai-app"
echo "   Stop all: docker-compose down"
echo "   Update: ./deploy.sh production 2"

# Setup monitoring dashboards
if [ "$ENVIRONMENT" = "production" ]; then
    echo ""
    echo "📊 Setting up monitoring dashboards..."
    
    # Wait for Grafana to be ready
    sleep 15
    
    # Import dashboards (would need actual dashboard JSON files)
    echo "📈 Monitoring dashboards will be available at http://localhost:3000"
fi

echo ""
echo "✨ Cherry AI Streamlit Platform is now running!"