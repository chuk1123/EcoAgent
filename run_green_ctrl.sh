#!/bin/bash
# This is the script that AgentBeats controller will use to start the green agent
# It must listen on $HOST and $AGENT_PORT

# Resolve symlinks to get actual script location
SOURCE="${BASH_SOURCE[0]}"
while [ -L "$SOURCE" ]; do
    DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"
    SOURCE="$(readlink "$SOURCE")"
    [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE"
done
SCRIPT_DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# Use the venv
source "${SCRIPT_DIR}/venv-ctrl/bin/activate"

cd "${SCRIPT_DIR}"

# Set PUBLIC_URL for the agent card - include controller proxy path
TUNNEL_URL_FILE="${SCRIPT_DIR}/.state/green/tunnel_url"
STATE_DIR="${SCRIPT_DIR}/.state/green"
if [ -f "$TUNNEL_URL_FILE" ]; then
    TUNNEL_URL=$(cat "$TUNNEL_URL_FILE" | tr -d '[:space:]')
    
    # Try to get AGENT_ID from controller or detect from .ab folder
    if [ -z "$AGENT_ID" ]; then
        # Fallback: find agent_id from .ab folder
        AGENT_ID=$(ls "${STATE_DIR}/.ab/agents/" 2>/dev/null | head -1)
    fi
    
    # If we have agent_id, include proxy path
    if [ ! -z "$AGENT_ID" ]; then
        export PUBLIC_URL="${TUNNEL_URL}/to_agent/${AGENT_ID}"
        export AGENT_URL="${TUNNEL_URL}/to_agent/${AGENT_ID}/"
    else
        export PUBLIC_URL="${TUNNEL_URL}"
        export AGENT_URL="${TUNNEL_URL}/"
    fi
fi

# Fallback if no tunnel URL
if [ -z "$AGENT_URL" ]; then
    export AGENT_URL="http://${HOST:-0.0.0.0}:${AGENT_PORT:-8000}/"
fi

echo "Starting green agent with AGENT_URL=$AGENT_URL"
python -m uvicorn ecoagent.green_agent.main:app --host ${HOST:-0.0.0.0} --port ${AGENT_PORT:-8000}
