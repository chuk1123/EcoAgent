#!/bin/bash
# =============================================================================
# Deploy White Agent with AgentBeats Controller + Cloudflare Tunnel
# All-in-one script - Deploys BOTH agents (Linear Regression + Naive Last Value)
# =============================================================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}"

CTRL_PORT=8011
TUNNEL_LOG="/tmp/cloudflare_tunnel_white_ctrl.log"

echo "🚀 Deploying White Agents with AgentBeats Controller..."
echo "   - Linear Regression Agent"
echo "   - Naive Last Value Agent"

# Cleanup function
cleanup() {
    echo ""
    echo "🛑 Shutting down..."
    if [ ! -z "$TUNNEL_PID" ]; then
        kill $TUNNEL_PID 2>/dev/null || true
    fi
    if [ ! -z "$CTRL_PID" ]; then
        kill $CTRL_PID 2>/dev/null || true
    fi
    rm -f "$TUNNEL_LOG"
    exit 0
}
trap cleanup SIGINT SIGTERM

# Clean up stale agent state to prevent errors
rm -rf "${SCRIPT_DIR}/.ab"

# Use the venv with earthshaker
source "${SCRIPT_DIR}/venv-ctrl/bin/activate"

# Set environment variables
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# Create symlink for run.sh that agentbeats expects
ln -sf run_white_ctrl.sh run.sh
chmod +x run.sh

# Start cloudflared tunnel FIRST to get the URL
echo "🌐 Starting Cloudflare tunnel..."
cloudflared tunnel --url http://localhost:$CTRL_PORT > "$TUNNEL_LOG" 2>&1 &
TUNNEL_PID=$!

# Wait for tunnel URL
echo "⏳ Waiting for tunnel URL..."
TUNNEL_URL=""
for i in {1..30}; do
    if [ -f "$TUNNEL_LOG" ]; then
        TUNNEL_URL=$(grep -oE 'https://[a-zA-Z0-9-]+\.trycloudflare\.com' "$TUNNEL_LOG" | head -1 | tr -d '[:space:]')
        if [ ! -z "$TUNNEL_URL" ]; then
            break
        fi
    fi
    sleep 1
done

if [ -z "$TUNNEL_URL" ]; then
    echo "❌ Failed to get tunnel URL after 30 seconds"
    cleanup
    exit 1
fi

# Extract hostname for CLOUDRUN_HOST
TUNNEL_HOST=$(echo "$TUNNEL_URL" | sed 's|https://||' | tr -d '[:space:]')

echo "🔗 Tunnel URL: $TUNNEL_URL"

# Set CLOUDRUN_HOST so controller knows public URL
export CLOUDRUN_HOST="$TUNNEL_HOST"
export HTTPS_ENABLED=true

# Start the AgentBeats controller on different port
echo "📦 Starting AgentBeats Controller on port $CTRL_PORT..."
PORT=$CTRL_PORT "${SCRIPT_DIR}/venv-ctrl/bin/agentbeats" run_ctrl &
CTRL_PID=$!

# Wait for controller to start
sleep 5

# Check if controller is running
if ! kill -0 $CTRL_PID 2>/dev/null; then
    echo "❌ Controller failed to start!"
    cleanup
    exit 1
fi

echo ""
echo "=============================================="
echo "✅ White Agents Controller Deployed!"
echo "=============================================="
echo ""
echo "🤖 Available Agents:"
echo "   • linear-regression  - Linear Regression Agent"
echo "   • naive-last-value   - Naive Last Value Agent"
echo ""
echo "🔗 Controller URL: $TUNNEL_URL"
echo ""
echo "📝 Use this URL in AgentBeats as Controller URL"
echo ""
echo "Press Ctrl+C to stop..."
echo ""

# Keep running
wait $CTRL_PID
