#!/bin/bash

# Universal Engine Deployment Script
# Version: 1.0
# Description: Automated deployment script for various environments

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEPLOYMENT_LOG="$PROJECT_ROOT/deployment.log"

# Default values
ENVIRONMENT="development"
LLM_PROVIDER="ollama"
DEPLOYMENT_METHOD="docker-compose"
SKIP_TESTS="false"
SKIP_BACKUP="false"
FORCE_DEPLOY="false"
VERBOSE="false"

# Function to print colored output
print_status() { echo -e "${GREEN}[$(date '+%H:%M:%S')] $1${NC}" | tee -a "$DEPLOYMENT_LOG"; }
print_warning() { echo -e "${YELLOW}[$(date '+%H:%M:%S')] WARNING: $1${NC}" | tee -a "$DEPLOYMENT_LOG"; }
print_error() { echo -e "${RED}[$(date '+%H:%M:%S')] ERROR: $1${NC}" | tee -a "$DEPLOYMENT_LOG"; }
print_info() { echo -e "${BLUE}[$(date '+%H:%M:%S')] $1${NC}" | tee -a "$DEPLOYMENT_LOG"; }
print_debug() { [[ "$VERBOSE" == "true" ]] && echo -e "${PURPLE}[$(date '+%H:%M:%S')] DEBUG: $1${NC}" | tee -a "$DEPLOYMENT_LOG" || true; }

# Function to show usage
show_usage() {
    cat << EOF
Universal Engine Deployment Script

Usage: $0 [OPTIONS]

OPTIONS:
    -e, --env ENVIRONMENT       Target environment (development, staging, production)
    -m, --method METHOD         Deployment method (docker-compose, kubernetes, manual)
    -l, --llm-provider PROVIDER LLM provider (ollama, openai, anthropic)
    -s, --skip-tests           Skip running tests before deployment
    -b, --skip-backup          Skip creating backup before deployment
    -f, --force                Force deployment without confirmation
    -v, --verbose              Enable verbose output
    -h, --help                 Show this help message

EXAMPLES:
    $0 --env production --method kubernetes
    $0 --env staging --llm-provider openai --skip-tests
    $0 --env development --method docker-compose --verbose

ENVIRONMENT VARIABLES:
    OPENAI_API_KEY            OpenAI API key (required for openai provider)
    ANTHROPIC_API_KEY         Anthropic API key (required for anthropic provider)
    POSTGRES_PASSWORD         PostgreSQL password for production
    GRAFANA_PASSWORD          Grafana admin password
    REGISTRY_URL              Docker registry URL for image pushing
    KUBECONFIG               Kubernetes config file path

EOF
}

# Function to parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -e|--env)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -m|--method)
                DEPLOYMENT_METHOD="$2"
                shift 2
                ;;
            -l|--llm-provider)
                LLM_PROVIDER="$2"
                shift 2
                ;;
            -s|--skip-tests)
                SKIP_TESTS="true"
                shift
                ;;
            -b|--skip-backup)
                SKIP_BACKUP="true"
                shift
                ;;
            -f|--force)
                FORCE_DEPLOY="true"
                shift
                ;;
            -v|--verbose)
                VERBOSE="true"
                shift
                ;;
            -h|--help)
                show_usage
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
}

# Function to validate environment
validate_environment() {
    print_info "Validating deployment environment..."
    
    # Check if environment is valid
    case $ENVIRONMENT in
        development|staging|production)
            print_debug "Environment '$ENVIRONMENT' is valid"
            ;;
        *)
            print_error "Invalid environment: $ENVIRONMENT"
            print_error "Valid environments: development, staging, production"
            exit 1
            ;;
    esac
    
    # Check deployment method
    case $DEPLOYMENT_METHOD in
        docker-compose|kubernetes|manual)
            print_debug "Deployment method '$DEPLOYMENT_METHOD' is valid"
            ;;
        *)
            print_error "Invalid deployment method: $DEPLOYMENT_METHOD"
            print_error "Valid methods: docker-compose, kubernetes, manual"
            exit 1
            ;;
    esac
    
    # Check LLM provider
    case $LLM_PROVIDER in
        ollama|openai|anthropic)
            print_debug "LLM provider '$LLM_PROVIDER' is valid"
            ;;
        *)
            print_error "Invalid LLM provider: $LLM_PROVIDER"
            print_error "Valid providers: ollama, openai, anthropic"
            exit 1
            ;;
    esac
    
    # Check required tools
    local required_tools=("git" "curl")
    
    if [[ "$DEPLOYMENT_METHOD" == "docker-compose" ]]; then
        required_tools+=("docker" "docker-compose")
    elif [[ "$DEPLOYMENT_METHOD" == "kubernetes" ]]; then
        required_tools+=("kubectl" "helm")
    fi
    
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            print_error "Required tool '$tool' is not installed"
            exit 1
        fi
    done
    
    # Check API keys if needed
    if [[ "$LLM_PROVIDER" == "openai" && -z "${OPENAI_API_KEY:-}" ]]; then
        print_error "OPENAI_API_KEY environment variable is required for OpenAI provider"
        exit 1
    fi
    
    if [[ "$LLM_PROVIDER" == "anthropic" && -z "${ANTHROPIC_API_KEY:-}" ]]; then
        print_error "ANTHROPIC_API_KEY environment variable is required for Anthropic provider"
        exit 1
    fi
    
    print_status "Environment validation completed"
}

# Function to run pre-deployment tests
run_tests() {
    if [[ "$SKIP_TESTS" == "true" ]]; then
        print_warning "Skipping tests as requested"
        return 0
    fi
    
    print_info "Running pre-deployment tests..."
    
    cd "$PROJECT_ROOT"
    
    # Activate virtual environment if it exists
    if [[ -f ".venv/bin/activate" ]]; then
        source .venv/bin/activate
        print_debug "Activated Python virtual environment"
    fi
    
    # Install test dependencies
    if [[ -f "requirements-dev.txt" ]]; then
        pip install -r requirements-dev.txt > /dev/null 2>&1
        print_debug "Installed development dependencies"
    fi
    
    # Run linting
    if command -v ruff &> /dev/null; then
        print_info "Running code linting..."
        ruff check . || {
            print_error "Linting failed"
            return 1
        }
        print_status "Linting passed"
    fi
    
    # Run type checking
    if command -v mypy &> /dev/null; then
        print_info "Running type checking..."
        mypy core/ tests/ || {
            print_warning "Type checking found issues (non-blocking)"
        }
    fi
    
    # Run unit tests
    print_info "Running unit tests..."
    python -m pytest tests/ -v --tb=short --disable-warnings || {
        print_error "Unit tests failed"
        return 1
    }
    
    # Run integration tests
    if [[ -d "tests/integration" ]]; then
        print_info "Running integration tests..."
        python -m pytest tests/integration/ -v --tb=short --disable-warnings || {
            print_error "Integration tests failed"
            return 1
        }
    fi
    
    # Run security tests
    if [[ -f "tests/test_security_privacy_verification.py" ]]; then
        print_info "Running security tests..."
        python tests/test_security_privacy_verification.py || {
            print_warning "Some security tests failed (review required)"
        }
    fi
    
    print_status "All tests completed successfully"
}

# Function to create backup
create_backup() {
    if [[ "$SKIP_BACKUP" == "true" ]]; then
        print_warning "Skipping backup as requested"
        return 0
    fi
    
    print_info "Creating deployment backup..."
    
    local backup_dir="$PROJECT_ROOT/backups/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$backup_dir"
    
    # Backup configuration files
    cp -r "$PROJECT_ROOT"/{.env*,docker-compose*.yml,*.json,*.yaml} "$backup_dir/" 2>/dev/null || true
    
    # Backup database if exists
    if [[ -f "$PROJECT_ROOT/data/universal_engine.db" ]]; then
        cp "$PROJECT_ROOT/data/universal_engine.db" "$backup_dir/"
        print_debug "Database backup created"
    fi
    
    # Backup logs
    if [[ -d "$PROJECT_ROOT/logs" ]]; then
        cp -r "$PROJECT_ROOT/logs" "$backup_dir/"
        print_debug "Logs backup created"
    fi
    
    # Create backup manifest
    cat > "$backup_dir/manifest.json" << EOF
{
    "backup_timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "environment": "$ENVIRONMENT",
    "llm_provider": "$LLM_PROVIDER",
    "deployment_method": "$DEPLOYMENT_METHOD",
    "git_commit": "$(git rev-parse HEAD 2>/dev/null || echo 'unknown')",
    "git_branch": "$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'unknown')"
}
EOF
    
    print_status "Backup created at: $backup_dir"
}

# Function to build Docker images
build_docker_images() {
    if [[ "$DEPLOYMENT_METHOD" != "docker-compose" && "$DEPLOYMENT_METHOD" != "kubernetes" ]]; then
        return 0
    fi
    
    print_info "Building Docker images..."
    
    cd "$PROJECT_ROOT"
    
    # Build main application image
    local image_tag="universal-engine:$(date +%Y%m%d-%H%M%S)"
    if [[ -n "${REGISTRY_URL:-}" ]]; then
        image_tag="${REGISTRY_URL}/universal-engine:$(date +%Y%m%d-%H%M%S)"
    fi
    
    docker build \
        --build-arg ENVIRONMENT="$ENVIRONMENT" \
        --build-arg LLM_PROVIDER="$LLM_PROVIDER" \
        -t "$image_tag" \
        -t "universal-engine:latest" \
        . || {
        print_error "Docker image build failed"
        return 1
    }
    
    print_status "Docker image built: $image_tag"
    
    # Push to registry if configured
    if [[ -n "${REGISTRY_URL:-}" ]]; then
        print_info "Pushing image to registry..."
        docker push "$image_tag" || {
            print_error "Failed to push image to registry"
            return 1
        }
        print_status "Image pushed to registry"
    fi
}

# Function to deploy with Docker Compose
deploy_docker_compose() {
    print_info "Deploying with Docker Compose..."
    
    cd "$PROJECT_ROOT"
    
    # Create environment-specific compose file
    local compose_file="docker-compose.${ENVIRONMENT}.yml"
    if [[ ! -f "$compose_file" ]]; then
        compose_file="docker-compose.yml"
    fi
    
    # Set environment variables
    export LLM_PROVIDER
    export ENVIRONMENT
    
    # Pull latest images
    print_info "Pulling latest images..."
    docker-compose -f "$compose_file" pull || print_warning "Some images could not be pulled"
    
    # Stop existing services
    print_info "Stopping existing services..."
    docker-compose -f "$compose_file" down --remove-orphans
    
    # Start services
    print_info "Starting services..."
    docker-compose -f "$compose_file" up -d || {
        print_error "Failed to start services with Docker Compose"
        return 1
    }
    
    # Wait for services to be ready
    print_info "Waiting for services to be ready..."
    sleep 30
    
    # Check service health
    local services=("universal-engine" "ollama" "redis")
    for service in "${services[@]}"; do
        if docker-compose -f "$compose_file" ps "$service" | grep -q "Up"; then
            print_status "$service is running"
        else
            print_error "$service failed to start"
            docker-compose -f "$compose_file" logs "$service"
            return 1
        fi
    done
    
    print_status "Docker Compose deployment completed"
}

# Function to deploy with Kubernetes
deploy_kubernetes() {
    print_info "Deploying with Kubernetes..."
    
    cd "$PROJECT_ROOT"
    
    # Check if kubectl is configured
    if ! kubectl cluster-info &> /dev/null; then
        print_error "kubectl is not properly configured or cluster is not accessible"
        return 1
    fi
    
    # Create namespace if it doesn't exist
    kubectl create namespace "universal-engine-$ENVIRONMENT" --dry-run=client -o yaml | kubectl apply -f -
    
    # Apply Kubernetes manifests
    local k8s_dir="k8s/$ENVIRONMENT"
    if [[ -d "$k8s_dir" ]]; then
        print_info "Applying Kubernetes manifests from $k8s_dir..."
        kubectl apply -f "$k8s_dir/" -n "universal-engine-$ENVIRONMENT" || {
            print_error "Failed to apply Kubernetes manifests"
            return 1
        }
    else
        print_warning "No Kubernetes manifests found for environment $ENVIRONMENT"
    fi
    
    # Wait for deployment to be ready
    print_info "Waiting for deployment to be ready..."
    kubectl wait --for=condition=available --timeout=300s deployment/universal-engine -n "universal-engine-$ENVIRONMENT" || {
        print_error "Deployment did not become ready within 5 minutes"
        kubectl describe deployment universal-engine -n "universal-engine-$ENVIRONMENT"
        return 1
    }
    
    # Get service information
    kubectl get services -n "universal-engine-$ENVIRONMENT"
    
    print_status "Kubernetes deployment completed"
}

# Function to perform manual deployment
deploy_manual() {
    print_info "Performing manual deployment..."
    
    cd "$PROJECT_ROOT"
    
    # Stop existing services
    if [[ -f "scripts/stop_universal_engine.sh" ]]; then
        print_info "Stopping existing services..."
        bash scripts/stop_universal_engine.sh --skip-verify || print_warning "Stop script had issues"
    fi
    
    # Update dependencies
    print_info "Installing/updating dependencies..."
    pip install -r requirements.txt
    if [[ -f "requirements-prod.txt" ]]; then
        pip install -r requirements-prod.txt
    fi
    
    # Run database migrations if needed
    if [[ -f "alembic.ini" ]]; then
        print_info "Running database migrations..."
        alembic upgrade head || print_warning "Database migration had issues"
    fi
    
    # Start services
    if [[ -f "scripts/start_universal_engine.sh" ]]; then
        print_info "Starting Universal Engine services..."
        bash scripts/start_universal_engine.sh || {
            print_error "Failed to start services"
            return 1
        }
    else
        print_error "Startup script not found"
        return 1
    fi
    
    print_status "Manual deployment completed"
}

# Function to run post-deployment tests
run_post_deployment_tests() {
    print_info "Running post-deployment health checks..."
    
    local base_url="http://localhost:8000"
    if [[ "$DEPLOYMENT_METHOD" == "kubernetes" ]]; then
        # Get the service URL for Kubernetes
        local service_info
        service_info=$(kubectl get service universal-engine-service -n "universal-engine-$ENVIRONMENT" -o jsonpath='{.status.loadBalancer.ingress[0].ip}:{.spec.ports[0].port}' 2>/dev/null || echo "")
        if [[ -n "$service_info" ]]; then
            base_url="http://$service_info"
        fi
    fi
    
    # Test health endpoint
    print_info "Testing health endpoint..."
    local attempts=0
    local max_attempts=10
    
    while [[ $attempts -lt $max_attempts ]]; do
        if curl -sf "$base_url/health" > /dev/null; then
            print_status "Health check passed"
            break
        fi
        
        attempts=$((attempts + 1))
        if [[ $attempts -eq $max_attempts ]]; then
            print_error "Health check failed after $max_attempts attempts"
            return 1
        fi
        
        print_debug "Health check attempt $attempts/$max_attempts failed, retrying in 10 seconds..."
        sleep 10
    done
    
    # Test basic functionality
    print_info "Testing basic functionality..."
    local test_response
    test_response=$(curl -sf "$base_url/api/v1/status" 2>/dev/null || echo "")
    
    if [[ -n "$test_response" ]]; then
        print_status "Basic functionality test passed"
    else
        print_warning "Basic functionality test failed (non-critical)"
    fi
    
    print_status "Post-deployment tests completed"
}

# Function to setup monitoring
setup_monitoring() {
    if [[ "$ENVIRONMENT" == "development" ]]; then
        print_info "Skipping monitoring setup for development environment"
        return 0
    fi
    
    print_info "Setting up monitoring..."
    
    case $DEPLOYMENT_METHOD in
        docker-compose)
            # Start monitoring stack
            if [[ -f "docker-compose.monitoring.yml" ]]; then
                docker-compose -f docker-compose.monitoring.yml up -d prometheus grafana
                print_status "Monitoring stack started with Docker Compose"
            fi
            ;;
        kubernetes)
            # Deploy monitoring manifests
            if [[ -d "k8s/monitoring" ]]; then
                kubectl apply -f k8s/monitoring/ -n "universal-engine-$ENVIRONMENT"
                print_status "Monitoring deployed with Kubernetes"
            fi
            ;;
        manual)
            print_warning "Manual monitoring setup required"
            ;;
    esac
}

# Function to show deployment summary
show_deployment_summary() {
    print_info "Deployment Summary"
    echo "==================="
    echo "Environment: $ENVIRONMENT"
    echo "LLM Provider: $LLM_PROVIDER"
    echo "Deployment Method: $DEPLOYMENT_METHOD"
    echo "Timestamp: $(date)"
    echo "Git Commit: $(git rev-parse HEAD 2>/dev/null || echo 'unknown')"
    echo ""
    
    case $DEPLOYMENT_METHOD in
        docker-compose)
            echo "Service URLs:"
            echo "  - Universal Engine: http://localhost:8000"
            echo "  - Grafana: http://localhost:3000 (admin/admin)"
            echo "  - Prometheus: http://localhost:9090"
            ;;
        kubernetes)
            echo "Use 'kubectl get services -n universal-engine-$ENVIRONMENT' to see service endpoints"
            ;;
        manual)
            echo "Service URL: http://localhost:8000"
            ;;
    esac
    
    echo ""
    echo "Logs: $DEPLOYMENT_LOG"
    echo "Backup: $(ls -t "$PROJECT_ROOT/backups/" 2>/dev/null | head -1 || echo 'None')"
}

# Function to handle cleanup on failure
cleanup_on_failure() {
    print_error "Deployment failed, performing cleanup..."
    
    case $DEPLOYMENT_METHOD in
        docker-compose)
            docker-compose down --remove-orphans 2>/dev/null || true
            ;;
        kubernetes)
            kubectl delete namespace "universal-engine-$ENVIRONMENT" --ignore-not-found=true
            ;;
        manual)
            if [[ -f "scripts/stop_universal_engine.sh" ]]; then
                bash scripts/stop_universal_engine.sh --force
            fi
            ;;
    esac
}

# Main deployment function
main() {
    echo -e "${BLUE}🚀 Universal Engine Deployment Script${NC}"
    echo "====================================="
    
    # Initialize deployment log
    echo "Deployment started at $(date)" > "$DEPLOYMENT_LOG"
    
    # Parse command line arguments
    parse_args "$@"
    
    print_info "Starting deployment with following configuration:"
    print_info "  Environment: $ENVIRONMENT"
    print_info "  LLM Provider: $LLM_PROVIDER"
    print_info "  Deployment Method: $DEPLOYMENT_METHOD"
    
    # Confirmation prompt (unless forced)
    if [[ "$FORCE_DEPLOY" != "true" ]]; then
        echo ""
        read -p "Do you want to proceed with this deployment? (y/N): " -n 1 -r
        echo ""
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_info "Deployment cancelled by user"
            exit 0
        fi
    fi
    
    # Set trap for cleanup on failure
    trap cleanup_on_failure ERR
    
    # Execute deployment steps
    validate_environment
    create_backup
    run_tests
    build_docker_images
    
    case $DEPLOYMENT_METHOD in
        docker-compose)
            deploy_docker_compose
            ;;
        kubernetes)
            deploy_kubernetes
            ;;
        manual)
            deploy_manual
            ;;
    esac
    
    run_post_deployment_tests
    setup_monitoring
    
    # Remove error trap on success
    trap - ERR
    
    print_status "🎉 Deployment completed successfully!"
    show_deployment_summary
}

# Run main function
main "$@"