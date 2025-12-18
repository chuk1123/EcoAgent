#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Activate conda environment (if using conda)
if command -v conda &> /dev/null; then
    source $(conda info --base)/etc/profile.d/conda.sh
    # Activate your environment - change 'constraintbench' to your env name if different
    conda activate constraintbench 2>/dev/null || echo "Conda env not found, using system Python"
fi

# Set environment variables
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"
export PORT=${AGENT_PORT:-8001}
export GREEN_AGENT_URL=${GREEN_AGENT_URL:-"http://localhost:8000"}
export HOST=${HOST:-0.0.0.0}

# Run white agent (local dev on port 8001)
cd "${SCRIPT_DIR}"
uvicorn ecoagent.white_agents.main:app --host ${HOST} --port ${PORT}