#!/bin/bash

# =============================================================================
# Deploy Green Agent with Cloudflare Tunnel
# Automatically captures the tunnel URL and sets PUBLIC_URL
# =============================================================================

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PORT=${PORT:-9000}
TUNNEL_LOG="/tmp/cloudflare_tunnel_green.log"

echo "🚀 Deploying Green Agent..."
echo "   Port: $PORT"

# Cleanup function
cleanup() {
    echo ""
    echo "🛑 Shutting down..."
    if [ ! -z "$TUNNEL_PID" ]; then
        kill $TUNNEL_PID 2>/dev/null || true
    fi
    if [ ! -z "$SERVER_PID" ]; then
        kill $SERVER_PID 2>/dev/null || true
    fi
    rm -f "$TUNNEL_LOG"
    exit 0
}
trap cleanup SIGINT SIGTERM

# Activate conda environment (if using conda)
if command -v conda &> /dev/null; then
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate constraintbench 2>/dev/null || true
fi

# Set base environment variables
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# Start cloudflared tunnel FIRST to get the URL
echo "🌐 Starting Cloudflare tunnel..."
cloudflared tunnel --url http://localhost:$PORT > "$TUNNEL_LOG" 2>&1 &
TUNNEL_PID=$!

# Wait for tunnel URL to appear
echo "⏳ Waiting for tunnel URL..."
TUNNEL_URL=""
for i in {1..30}; do
    if [ -f "$TUNNEL_LOG" ]; then
        # Extract URL from cloudflared output (handles different output formats)
        TUNNEL_URL=$(grep -oE 'https://[a-zA-Z0-9-]+\.trycloudflare\.com' "$TUNNEL_LOG" | head -1 | tr -d '[:space:]')
        if [ ! -z "$TUNNEL_URL" ]; then
            break
        fi
    fi
    sleep 1
done

if [ -z "$TUNNEL_URL" ]; then
    echo "❌ Failed to get tunnel URL after 30 seconds"
    echo "   Check $TUNNEL_LOG for errors"
    cleanup
    exit 1
fi

# Extract just the hostname (without https://)
TUNNEL_HOST=$(echo "$TUNNEL_URL" | sed 's|https://||' | tr -d '[:space:]')

# Set PUBLIC_URL for the agent card
export PUBLIC_URL="$TUNNEL_URL"

echo "🔗 Tunnel URL acquired: $TUNNEL_URL"

# NOW start the FastAPI server with PUBLIC_URL set
echo "📦 Starting Green Agent server on port $PORT..."
cd "${SCRIPT_DIR}"
python -m uvicorn ecoagent.green_agent.main:app --host 0.0.0.0 --port $PORT &
SERVER_PID=$!

# Wait for server to be ready
echo "⏳ Waiting for server to start..."
sleep 3

# Check if server is running
if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "❌ Server failed to start!"
    cleanup
    exit 1
fi

echo ""
echo "=============================================="
echo "✅ Green Agent Deployed Successfully!"
echo "=============================================="
echo ""
echo "🔗 Tunnel URL: $TUNNEL_URL"
echo ""
echo "📋 Agent Card Endpoints:"
echo "   $TUNNEL_URL/.well-known/agent.json      (A2A standard)"
echo "   $TUNNEL_URL/.well-known/agent-card.json (legacy)"
echo ""
echo "🧪 Test with:"
echo "   curl $TUNNEL_URL/health"
echo "   curl $TUNNEL_URL/status"
echo "   curl $TUNNEL_URL/.well-known/agent.json"
echo ""
echo "📝 For AgentBeats, use Controller URL:"
echo "   $TUNNEL_URL"
echo ""
echo "Press Ctrl+C to stop..."
echo ""

# Keep running until interrupted
wait $SERVER_PID
