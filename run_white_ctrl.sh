#!/bin/bash
# This is the script that AgentBeats controller will use to start the white agent
# It must listen on $HOST and $AGENT_PORT

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# Use the venv
source "${SCRIPT_DIR}/venv-ctrl/bin/activate"

cd "${SCRIPT_DIR}"
python -m uvicorn ecoagent.white_agents.main:app --host ${HOST:-0.0.0.0} --port ${AGENT_PORT:-8000}

