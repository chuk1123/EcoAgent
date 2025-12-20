import os
import sys
import json
import uuid
import logging
import pandas as pd

from ecoagent.green_agent.green_agent import GreenAgent
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [GREEN AGENT] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

app = FastAPI()

# Get the public URL from environment (set by deploy script or manually)
PUBLIC_URL = os.getenv("PUBLIC_URL", "http://localhost:9000")

# Add CORS middleware for AgentBeats platform
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

green_agent = GreenAgent("ecoagent/contexts/housing")


def process_command(command_text: str) -> str:
    """Process a command and return the result as a string."""
    try:
        command = json.loads(command_text)
    except:
        logger.warning(
            f"<<< RECEIVED INVALID JSON from WHITE AGENT: {command_text[:200]}"
        )
        return json.dumps({"error": "Invalid JSON format"})

    action = command.get("action")
    args = command.get("args", {})

    # Log what we received from white agent
    logger.info(f"<<< RECEIVED FROM WHITE AGENT: action={action}, args={args}")

    if action == "describe":
        info = green_agent.describe()
        result = json.dumps(info)
        logger.info(
            f">>> SENDING TO WHITE AGENT: describe info - context={info.get('context')}, target={info.get('target')}"
        )
        return result

    elif action == "request_dataset":
        ds_id = command["args"]["ds_id"]
        split = command["args"].get("split", "train")
        try:
            df = green_agent.request_dataset(ds_id, split)
            result = df.to_json(orient="split")
            logger.info(
                f">>> SENDING TO WHITE AGENT: dataset '{ds_id}' ({split}) - {len(df)} rows, columns={list(df.columns)}"
            )
            return result
        except Exception as e:
            logger.error(f">>> SENDING TO WHITE AGENT: error - {str(e)}")
            return json.dumps({"error": str(e)})

    elif action == "evaluate_predictions":
        y_pred = command["args"]["y_pred"]
        result = green_agent.evaluate_predictions(y_pred)
        logger.info(
            f">>> SENDING TO WHITE AGENT: evaluation - rmse={result.get('rmse')}, final_score={result.get('final_score')}, datasets_used={result.get('datasets_used')}"
        )
        return json.dumps(result)

    logger.warning(f">>> SENDING TO WHITE AGENT: error - Unknown action '{action}'")
    return json.dumps({"error": "Unknown action"})


# =============================================================================
# A2A JSON-RPC Endpoint (POST to root)
# =============================================================================
@app.post("/")
async def a2a_jsonrpc_handler(request: Request):
    """Handle A2A JSON-RPC messages at the root URL."""
    try:
        data = await request.json()
    except:
        return JSONResponse(
            content={
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": "Parse error"},
            },
            status_code=400,
        )

    # Extract JSON-RPC fields
    jsonrpc = data.get("jsonrpc", "2.0")
    request_id = data.get("id")
    method = data.get("method", "")
    params = data.get("params", {})

    # Handle message/send method (A2A protocol)
    if method == "message/send":
        # Extract text from A2A message format
        message = params.get("message", {})
        parts = message.get("parts", [])

        # Find text content
        text_content = ""
        for part in parts:
            if part.get("kind") == "text":
                text_content = part.get("text", "")
                break

        # Process the command
        result_text = process_command(text_content)

        # Return A2A JSON-RPC response
        return JSONResponse(
            content={
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "message": {
                        "messageId": str(uuid.uuid4()),
                        "role": "agent",
                        "parts": [{"kind": "text", "text": result_text}],
                    }
                },
            }
        )

    # Unknown method
    return JSONResponse(
        content={
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        },
        status_code=400,
    )


# =============================================================================
# OpenAI-style chat completions (legacy)
# =============================================================================
@app.post("/v1/chat/completions")
async def interaction_handler(request: Request):
    data = await request.json()
    messages = data.get("messages", [])
    last_message = messages[-1]["content"]
    result = process_command(last_message)
    return {"choices": [{"message": {"content": result}}]}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/status")
def status():
    return {"status": "ok"}


def get_agent_card():
    """Return the agent card in A2A protocol format."""
    return {
        "name": "EcoAgent Green Agent",
        "version": "1.0.0",
        "protocolVersion": "0.3.0",
        "description": "Green Agent evaluator for EcoAgent (A2A). Evaluates white agents on resource-constrained housing forecasting tasks.",
        "url": PUBLIC_URL + "/",
        "defaultInputModes": ["text"],
        "defaultOutputModes": ["text"],
        "skills": [],
        "capabilities": {
            "streaming": False,
            "pushNotifications": False,
            "stateTransitionHistory": False,
        },
    }


# A2A Protocol standard path
@app.get("/.well-known/agent.json")
def agent_card_a2a():
    return JSONResponse(content=get_agent_card(), media_type="application/json")


# Legacy path for backwards compatibility
@app.get("/.well-known/agent-card.json")
def agent_card_legacy():
    return JSONResponse(content=get_agent_card(), media_type="application/json")


# AgentBeats controller endpoint - returns dict of agent instances
@app.get("/agents")
def list_agents():
    agent_id = "green-agent-housing"
    return {
        agent_id: {
            "id": agent_id,
            "name": "EcoAgent Green Agent",
            "url": PUBLIC_URL,
            "status": "ready",
            "ready": True,
            "agent_card": get_agent_card(),
        }
    }


# AgentBeats per-agent status endpoint
@app.get("/agents/{agent_id}")
def get_agent_status(agent_id: str):
    return {
        "id": agent_id,
        "name": "EcoAgent Green Agent",
        "url": PUBLIC_URL,
        "status": "ready",
        "ready": True,
        "agent_card": get_agent_card(),
    }


@app.post("/reset")
def reset_agent():
    green_agent.reset()
    return {"status": "state reset", "budget_restored": green_agent.budget.max}


# AgentBeats reset endpoint (per-agent path)
@app.post("/agents/{agent_id}/reset")
def reset_agent_by_id(agent_id: str):
    green_agent.reset()
    return {"status": "ok", "agent_id": agent_id, "message": "Agent reset successfully"}
