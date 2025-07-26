#!/usr/bin/env bash
set -euo pipefail

# Verbose Agent Diagnostics Script
# Provides comprehensive agent health analysis with detailed error reporting

# Dependency checks
need() { 
    command -v "$1" >/dev/null || { 
        echo "❌ Missing dependency: $1"; 
        exit 1; 
    } 
}
need jq
need curl  
need nc

JQ=jq

# Color codes for output formatting
RED='\e[31m'
GRN='\e[32m'
YLW='\e[33m'
BLU='\e[34m'
MAG='\e[35m'
CYN='\e[36m'
NC='\e[0m' # No Color

# Configuration
CONFIG_FILE="config/agents.json"
TIMEOUT=5
VERBOSE=true

echo -e "${CYN}🔍 Cherry AI Platform - Verbose Agent Diagnostics${NC}"
echo -e "${CYN}=================================================${NC}"
echo

# Check if config file exists
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo -e "${RED}❌ Config file not found: $CONFIG_FILE${NC}"
    exit 1
fi

# Validate JSON format
if ! $JQ empty "$CONFIG_FILE" 2>/dev/null; then
    echo -e "${RED}❌ Invalid JSON in config file: $CONFIG_FILE${NC}"
    exit 1
fi

# Global counters
total_agents=0
healthy_agents=0
critical_down=0

# Extract agent information and process each agent
$JQ -r '.agents[] | "\(.name)\t\(.endpoint)\t\(.enabled // true)\t\(.priority // 99)\t(.timeout // 300)"' "$CONFIG_FILE" | \
while IFS=$'\t' read -r name endpoint enabled priority timeout; do
    total_agents=$((total_agents + 1))
    
    echo -e "${BLU}🔧 Agent: ${name}${NC}"
    echo -e "   Endpoint: ${endpoint}"
    echo -e "   Enabled: ${enabled}, Priority: ${priority}, Timeout: ${timeout}s"
    
    # Skip disabled agents
    if [[ "$enabled" == "false" ]]; then
        echo -e "   ${YLW}⏸️  SKIPPED (disabled)${NC}"
        echo
        continue
    fi
    
    # Parse endpoint
    hostport="${endpoint#http://}"
    hostport="${hostport#https://}"
    host="${hostport%:*}"
    port="${hostport##*:}"
    
    echo -e "   Host: ${host}, Port: ${port}"
    
    # Test 1: Port connectivity
    echo -e "   ${CYN}🔌 Port Connectivity Test${NC}"
    if nc -z "$host" "$port" 2>/dev/null; then
        echo -e "   ${GRN}✅ Port ${port} is open and accepting connections${NC}"
    else
        echo -e "   ${RED}❌ Port ${port} is closed or unreachable${NC}"
        echo -e "   ${YLW}💡 Troubleshooting: Check if agent process is running on port ${port}${NC}"
        echo
        continue
    fi
    
    # Test 2: Agent metadata endpoint
    echo -e "   ${CYN}📋 Agent Metadata Test (.well-known/agent.json)${NC}"
    metadata_url="${endpoint}/.well-known/agent.json"
    
    if metadata_response=$(curl -sS -m "$TIMEOUT" "$metadata_url" 2>&1); then
        if echo "$metadata_response" | $JQ empty 2>/dev/null; then
            agent_name=$(echo "$metadata_response" | $JQ -r '.name // "unknown"' 2>/dev/null)
            agent_version=$(echo "$metadata_response" | $JQ -r '.version // "unknown"' 2>/dev/null)
            agent_status=$(echo "$metadata_response" | $JQ -r '.status // "unknown"' 2>/dev/null)
            
            echo -e "   ${GRN}✅ Metadata endpoint accessible${NC}"
            echo -e "   📝 Name: ${agent_name}, Version: ${agent_version}, Status: ${agent_status}"
            
            # Display full metadata if verbose
            if [[ "$VERBOSE" == "true" ]]; then
                echo -e "   ${MAG}📄 Full metadata:${NC}"
                echo "$metadata_response" | $JQ . 2>/dev/null | sed 's/^/      /' || echo "      (Invalid JSON)"
            fi
        else
            echo -e "   ${YLW}⚠️  Metadata endpoint returned invalid JSON${NC}"
            echo -e "   ${YLW}📄 Raw response: ${metadata_response}${NC}"
        fi
    else
        echo -e "   ${RED}❌ Metadata endpoint failed${NC}"
        echo -e "   ${RED}🚨 Error: ${metadata_response}${NC}"
        echo -e "   ${YLW}💡 Troubleshooting: Check if agent supports .well-known/agent.json endpoint${NC}"
    fi
    
    # Test 3: JSON-RPC health check
    echo -e "   ${CYN}🩺 JSON-RPC Health Check${NC}"
    health_payload='{"jsonrpc":"2.0","id":"health_check","method":"health_check","params":{}}'
    
    if health_response=$(curl -sS -m "$TIMEOUT" "$endpoint" \
        -H 'Content-Type: application/json' \
        -d "$health_payload" 2>&1); then
        
        if echo "$health_response" | $JQ empty 2>/dev/null; then
            rpc_result=$(echo "$health_response" | $JQ -r '.result // empty' 2>/dev/null)
            rpc_error=$(echo "$health_response" | $JQ -r '.error // empty' 2>/dev/null)
            
            if [[ -n "$rpc_result" ]]; then
                echo -e "   ${GRN}✅ JSON-RPC health check successful${NC}"
                echo -e "   💚 Result: ${rpc_result}"
                healthy_agents=$((healthy_agents + 1))
            elif [[ -n "$rpc_error" ]]; then
                echo -e "   ${YLW}⚠️  JSON-RPC returned error${NC}"
                echo -e "   ${YLW}🚨 Error: ${rpc_error}${NC}"
            else
                echo -e "   ${YLW}⚠️  JSON-RPC response unclear${NC}"
                echo -e "   ${YLW}📄 Raw response: ${health_response}${NC}"
            fi
            
            # Display full JSON-RPC response if verbose
            if [[ "$VERBOSE" == "true" ]]; then
                echo -e "   ${MAG}📄 Full JSON-RPC response:${NC}"
                echo "$health_response" | $JQ . 2>/dev/null | sed 's/^/      /' || echo "      (Invalid JSON)"
            fi
        else
            echo -e "   ${YLW}⚠️  JSON-RPC returned invalid JSON${NC}"
            echo -e "   ${YLW}📄 Raw response: ${health_response}${NC}"
        fi
    else
        echo -e "   ${RED}❌ JSON-RPC health check failed${NC}"
        echo -e "   ${RED}🚨 Error: ${health_response}${NC}"
        echo -e "   ${YLW}💡 Troubleshooting: Check if agent supports JSON-RPC protocol${NC}"
        
        # Check if this is a critical agent
        if [[ "$priority" -le 2 ]]; then
            critical_down=$((critical_down + 1))
        fi
    fi
    
    # Test 4: Additional endpoint probes
    echo -e "   ${CYN}🔍 Additional Endpoint Probes${NC}"
    
    # Try root endpoint
    if root_response=$(curl -sS -m "$TIMEOUT" "$endpoint" 2>&1); then
        if [[ ${#root_response} -gt 0 ]]; then
            echo -e "   ${GRN}✅ Root endpoint (/) responds${NC}"
            echo -e "   📄 Response length: ${#root_response} chars"
        else
            echo -e "   ${YLW}⚠️  Root endpoint responds but empty${NC}"
        fi
    else
        echo -e "   ${YLW}⚠️  Root endpoint (/) failed${NC}"
    fi
    
    # Try health endpoint
    if health_simple=$(curl -sS -m "$TIMEOUT" "${endpoint}/health" 2>&1); then
        echo -e "   ${GRN}✅ Simple health endpoint (/health) responds${NC}"
        echo -e "   📄 Response: ${health_simple:0:100}..."
    else
        echo -e "   ${YLW}⚠️  Simple health endpoint (/health) failed${NC}"
    fi
    
    echo
done

# Summary report
echo -e "${CYN}📊 DIAGNOSTIC SUMMARY${NC}"
echo -e "${CYN}===================${NC}"
echo -e "Total agents checked: ${total_agents}"
echo -e "Healthy agents: ${GRN}${healthy_agents}${NC}"
echo -e "Unhealthy agents: ${RED}$((total_agents - healthy_agents))${NC}"

if [[ "$critical_down" -gt 0 ]]; then
    echo -e "${RED}❌ Critical agents down: ${critical_down}${NC}"
    echo -e "${YLW}🚨 Platform may be degraded${NC}"
elif [[ "$healthy_agents" -eq "$total_agents" ]]; then
    echo -e "${GRN}✅ All agents are healthy${NC}"
    echo -e "${GRN}🎉 Platform is fully operational${NC}"
else
    echo -e "${YLW}⚠️  Some non-critical agents are down${NC}"
    echo -e "${YLW}💡 Platform is operational with reduced capacity${NC}"
fi

echo
echo -e "${CYN}💡 Troubleshooting Tips:${NC}"
echo -e "• Port closed: Check if agent process is running"
echo -e "• JSON-RPC failed: Verify agent supports JSON-RPC 2.0 protocol"
echo -e "• Metadata missing: Check if agent implements .well-known/agent.json"
echo -e "• Network issues: Verify firewall and network connectivity"
echo -e "• High priority agents (0-2) are critical for platform operation"

exit 0