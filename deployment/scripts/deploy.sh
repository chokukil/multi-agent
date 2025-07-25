#!/bin/bash

# Cherry AI Platform Deployment Script
# 검증된 배포 패턴과 모범 사례를 적용한 자동화된 배포 스크립트

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
DEPLOYMENT_TYPE="docker"
ENVIRONMENT="development"
VERSION="latest"
SKIP_TESTS=false
SKIP_BUILD=false
CLEANUP=false
VERBOSE=false

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Usage information
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Cherry AI Platform Deployment Script

OPTIONS:
    -t, --type TYPE         Deployment type: docker, kubernetes, local (default: docker)
    -e, --env ENV          Environment: development, staging, production (default: development)
    -v, --version VERSION  Version tag for deployment (default: latest)
    --skip-tests          Skip running tests before deployment
    --skip-build          Skip building Docker images
    --cleanup             Clean up previous deployments
    --verbose             Enable verbose output
    -h, --help            Show this help message

EXAMPLES:
    $0 --type docker --env production --version v1.0.0
    $0 --type kubernetes --env staging --cleanup
    $0 --type local --skip-tests

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--type)
            DEPLOYMENT_TYPE="$2"
            shift 2
            ;;
        -e|--env)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -v|--version)
            VERSION="$2"
            shift 2
            ;;
        --skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        --cleanup)
            CLEANUP=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Validate deployment type
if [[ ! "$DEPLOYMENT_TYPE" =~ ^(docker|kubernetes|local)$ ]]; then
    error "Invalid deployment type: $DEPLOYMENT_TYPE"
    exit 1
fi

# Validate environment
if [[ ! "$ENVIRONMENT" =~ ^(development|staging|production)$ ]]; then
    error "Invalid environment: $ENVIRONMENT"
    exit 1
fi

# Set script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Environment-specific configurations
set_environment_config() {
    case $ENVIRONMENT in
        development)
            export CHERRY_AI_DEBUG=true
            export CHERRY_AI_LOG_LEVEL=debug
            export CHERRY_AI_REPLICAS=1
            ;;
        staging)
            export CHERRY_AI_DEBUG=false
            export CHERRY_AI_LOG_LEVEL=info
            export CHERRY_AI_REPLICAS=2
            ;;
        production)
            export CHERRY_AI_DEBUG=false
            export CHERRY_AI_LOG_LEVEL=warning
            export CHERRY_AI_REPLICAS=3
            ;;
    esac
}

# Pre-deployment checks
pre_deployment_checks() {
    log "Running pre-deployment checks..."
    
    # Check if required tools are installed
    local required_tools=()
    
    case $DEPLOYMENT_TYPE in
        docker)
            required_tools+=("docker" "docker-compose")
            ;;
        kubernetes)
            required_tools+=("kubectl" "helm")
            ;;
        local)
            required_tools+=("python3" "pip")
            ;;
    esac
    
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            error "$tool is not installed or not in PATH"
            exit 1
        fi
    done
    
    # Check project structure
    if [[ ! -f "$PROJECT_ROOT/cherry_ai_streamlit_app.py" ]]; then
        error "Main application file not found"
        exit 1
    fi
    
    if [[ ! -f "$PROJECT_ROOT/requirements.txt" ]]; then
        error "Requirements file not found"
        exit 1
    fi
    
    success "Pre-deployment checks passed"
}

# Run tests
run_tests() {
    if [[ "$SKIP_TESTS" == true ]]; then
        warning "Skipping tests"
        return 0
    fi
    
    log "Running tests..."
    cd "$PROJECT_ROOT"
    
    # Check if tests exist
    if [[ -d "tests" ]]; then
        if command -v pytest &> /dev/null; then
            pytest tests/ -v
        else
            python -m unittest discover tests/
        fi
    else
        warning "No tests directory found, skipping tests"
    fi
    
    success "Tests completed"
}

# Build Docker images
build_docker_images() {
    if [[ "$SKIP_BUILD" == true ]]; then
        warning "Skipping Docker build"
        return 0
    fi
    
    log "Building Docker images..."
    cd "$PROJECT_ROOT"
    
    # Build main application image
    docker build \
        -t "cherry-ai/platform:$VERSION" \
        -t "cherry-ai/platform:latest" \
        -f deployment/docker/Dockerfile \
        .
    
    success "Docker images built successfully"
}

# Deploy with Docker Compose
deploy_docker() {
    log "Deploying with Docker Compose..."
    cd "$PROJECT_ROOT/deployment/docker"
    
    # Set environment variables
    export CHERRY_AI_VERSION="$VERSION"
    export CHERRY_AI_ENV="$ENVIRONMENT"
    
    if [[ "$CLEANUP" == true ]]; then
        log "Cleaning up previous deployment..."
        docker-compose down -v --remove-orphans
        docker system prune -f
    fi
    
    # Deploy services
    docker-compose up -d
    
    # Wait for services to be ready
    log "Waiting for services to be ready..."
    sleep 30
    
    # Health check
    if curl -f http://localhost:8501/_stcore/health &> /dev/null; then
        success "Cherry AI Platform is running at http://localhost:8501"
    else
        error "Health check failed"
        exit 1
    fi
}

# Deploy to Kubernetes
deploy_kubernetes() {
    log "Deploying to Kubernetes..."
    cd "$PROJECT_ROOT/deployment/kubernetes"
    
    # Check kubectl connectivity
    if ! kubectl cluster-info &> /dev/null; then
        error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    if [[ "$CLEANUP" == true ]]; then
        log "Cleaning up previous deployment..."
        kubectl delete namespace cherry-ai --ignore-not-found=true
        sleep 10
    fi
    
    # Apply Kubernetes manifests
    kubectl apply -f namespace.yaml
    kubectl apply -f configmap.yaml
    kubectl apply -f secrets.yaml
    kubectl apply -f pvc.yaml
    kubectl apply -f deployment.yaml
    kubectl apply -f ingress.yaml
    
    # Wait for rollout
    log "Waiting for deployment rollout..."
    kubectl rollout status deployment/cherry-ai-platform -n cherry-ai --timeout=300s
    
    # Get service info
    kubectl get services -n cherry-ai
    kubectl get ingress -n cherry-ai
    
    success "Kubernetes deployment completed"
}

# Local deployment
deploy_local() {
    log "Setting up local development environment..."
    cd "$PROJECT_ROOT"
    
    # Create virtual environment if it doesn't exist
    if [[ ! -d "venv" ]]; then
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Install dependencies
    pip install -r requirements.txt
    
    # Set environment variables
    export CHERRY_AI_ENV="$ENVIRONMENT"
    export CHERRY_AI_DEBUG=true
    
    # Run the application
    log "Starting Cherry AI Platform locally..."
    streamlit run cherry_ai_streamlit_app.py --server.port=8501 &
    
    # Store PID for cleanup
    echo $! > cherry_ai.pid
    
    sleep 10
    
    if curl -f http://localhost:8501/_stcore/health &> /dev/null; then
        success "Cherry AI Platform is running at http://localhost:8501"
        log "To stop the application, run: kill \$(cat cherry_ai.pid)"
    else
        error "Failed to start local application"
        exit 1
    fi
}

# Post-deployment tasks
post_deployment_tasks() {
    log "Running post-deployment tasks..."
    
    case $DEPLOYMENT_TYPE in
        docker)
            # Show running containers
            docker-compose ps
            
            # Show logs
            if [[ "$VERBOSE" == true ]]; then
                docker-compose logs --tail=50
            fi
            ;;
        kubernetes)
            # Show pod status
            kubectl get pods -n cherry-ai
            
            # Show logs
            if [[ "$VERBOSE" == true ]]; then
                kubectl logs -n cherry-ai -l app=cherry-ai-platform --tail=50
            fi
            ;;
        local)
            log "Local deployment completed"
            ;;
    esac
    
    success "Post-deployment tasks completed"
}

# Cleanup function
cleanup() {
    if [[ "$CLEANUP" != true ]]; then
        return 0
    fi
    
    log "Performing cleanup..."
    
    case $DEPLOYMENT_TYPE in
        docker)
            cd "$PROJECT_ROOT/deployment/docker"
            docker-compose down -v --remove-orphans
            docker system prune -f
            ;;
        kubernetes)
            kubectl delete namespace cherry-ai --ignore-not-found=true
            ;;
        local)
            if [[ -f "cherry_ai.pid" ]]; then
                kill "$(cat cherry_ai.pid)" 2>/dev/null || true
                rm cherry_ai.pid
            fi
            ;;
    esac
    
    success "Cleanup completed"
}

# Main execution
main() {
    log "Starting Cherry AI Platform deployment"
    log "Deployment type: $DEPLOYMENT_TYPE"
    log "Environment: $ENVIRONMENT"
    log "Version: $VERSION"
    
    # Set environment configuration
    set_environment_config
    
    # Run deployment steps
    pre_deployment_checks
    run_tests
    
    case $DEPLOYMENT_TYPE in
        docker)
            build_docker_images
            deploy_docker
            ;;
        kubernetes)
            build_docker_images
            deploy_kubernetes
            ;;
        local)
            deploy_local
            ;;
    esac
    
    post_deployment_tasks
    
    success "Cherry AI Platform deployment completed successfully!"
}

# Error handling
trap 'error "Deployment failed"; exit 1' ERR

# Run main function
main "$@"