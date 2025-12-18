import os
import sys
import json
import pandas as pd

from ecoagent.green_agent.green_agent import GreenAgent
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI()

# Get the public URL from environment (set by deploy script or manually)
PUBLIC_URL = os.getenv("PUBLIC_URL", "http://localhost:9000")

# Add CORS middleware for AgentBeats platform
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

green_agent = GreenAgent("ecoagent/contexts/housing")
active_sessions = {}

@app.post("/v1/chat/completions")
async def interaction_handler(request: Request):
    data = await request.json()
    messages = data.get("messages", [])
    last_message = messages[-1]["content"]
    
    try:
        command = json.loads(last_message)
    except:
        return {"choices": [{"message": {"content": "Error: Invalid JSON format"}}]}

    action = command.get("action")
    
    if action == "describe":
        info = green_agent.describe()
        return {"choices": [{"message": {"content": json.dumps(info)}}]}

    elif action == "request_dataset":
        ds_id = command["args"]["ds_id"]
        split = command["args"].get("split", "train")
        try:
            df = green_agent.request_dataset(ds_id, split)
            return {"choices": [{"message": {"content": df.to_json(orient="split")}}]}
        except Exception as e:
            return {"choices": [{"message": {"content": json.dumps({"error": str(e)})}}]}

    elif action == "evaluate_predictions":
        y_pred = command["args"]["y_pred"]
        result = green_agent.evaluate_predictions(y_pred)
        return {"choices": [{"message": {"content": json.dumps(result)}}]}

    return {"choices": [{"message": {"content": "Unknown action"}}]}

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
            "stateTransitionHistory": False
        }
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
            "agent_card": get_agent_card()
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
        "agent_card": get_agent_card()
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