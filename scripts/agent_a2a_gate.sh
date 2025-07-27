#!/bin/bash
# A2A Health Gate Script
# Checks if A2A agents are running on ports 8312 and 8315

echo "Checking A2A agents health..."

# Check port 8312 (EDA Tools)
if curl -s -o /dev/null -w "%{http_code}" http://localhost:8312/health 2>/dev/null | grep -q "200"; then
    echo "✅ A2A EDA Tools (8312) is healthy"
else
    echo "❌ A2A EDA Tools (8312) is not responding"
    echo "Please start the agent with: python -m a2a_agents.eda_tools --port 8312"
fi

# Check port 8315 (Pandas Hub)
if curl -s -o /dev/null -w "%{http_code}" http://localhost:8315/health 2>/dev/null | grep -q "200"; then
    echo "✅ A2A Pandas Hub (8315) is healthy"
else
    echo "❌ A2A Pandas Hub (8315) is not responding"
    echo "Please start the agent with: python -m a2a_agents.pandas_hub --port 8315"
fi

echo "A2A health check completed"