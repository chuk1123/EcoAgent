import os
import sys
import json
import requests
import pandas as pd
import numpy as np
import importlib
from importlib import import_module
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

class GreenAgentProxy:
    def __init__(self, green_agent_url):
        self.url = green_agent_url

    def _send_command(self, action, args=None):
        command_json = json.dumps({"action": action, "args": args or {}}, cls=NumpyEncoder)

        payload = {
            "messages": [
                {"role": "user", "content": command_json}
            ]
        }
        
        try:
            response = requests.post(f"{self.url}/v1/chat/completions", json=payload)
            response.raise_for_status()
            
            response_data = response.json()
            content = response_data["choices"][0]["message"]["content"]
            
            try:
                data = json.loads(content)
                if isinstance(data, dict) and "error" in data:
                    raise RuntimeError(data["error"])
                return data
            except json.JSONDecodeError:
                return content
        except Exception as e:
            raise RuntimeError(f"GreenAgent Communication Error: {e}")

    def describe(self):
        return self._send_command("describe")

    def request_dataset(self, ds_id, split="train"):
        json_data = self._send_command("request_dataset", {"ds_id": ds_id, "split": split})
        return pd.read_json(json.dumps(json_data), orient="split")

    def evaluate_predictions(self, y_pred):
        return self._send_command("evaluate_predictions", {"y_pred": y_pred})

app = FastAPI()

# Add CORS middleware for AgentBeats platform
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GREEN_AGENT_URL = os.getenv("GREEN_AGENT_URL")

# =============================================================================
# Two agents: Linear Regression and Naive Last Value (no LLM required)
# =============================================================================
AGENTS = {
    "linear-regression": {
        "module": "ecoagent.white_agents.two_feature_regression",
        "name": "Linear Regression Agent",
        "description": "Uses sklearn LinearRegression with median listing price as feature.",
    },
    "naive-last-value": {
        "module": "ecoagent.white_agents.naive_last_value",
        "name": "Naive Last Value Agent",
        "description": "Simple baseline that predicts the last observed training value.",
    },
}

def run_agent(agent_id: str, green_agent_url: str):
    """Run a specific agent and return results."""
    if agent_id not in AGENTS:
        raise ValueError(f"Unknown agent: {agent_id}")
    
    proxy = GreenAgentProxy(green_agent_url)
    agent_module = import_module(AGENTS[agent_id]["module"])
    importlib.reload(agent_module)
    return agent_module.run(proxy)

def get_agent_card(agent_id: str):
    """Generate agent card for a specific agent."""
    public_url = os.getenv("PUBLIC_URL", "http://localhost:9001")
    agent = AGENTS[agent_id]
    return {
        "name": agent["name"],
        "version": "1.0.0",
        "protocolVersion": "0.3.0",
        "description": f"White Agent for EcoAgent (A2A). {agent['description']}",
        "url": f"{public_url}/agents/{agent_id}/",
        "defaultInputModes": ["text"],
        "defaultOutputModes": ["text"],
        "skills": [],
        "capabilities": {
            "streaming": False,
            "pushNotifications": False,
            "stateTransitionHistory": False
        }
    }

# =============================================================================
# Per-agent chat completions endpoint
# =============================================================================
@app.post("/agents/{agent_id}/v1/chat/completions")
async def agent_chat(agent_id: str, request: Request):
    if agent_id not in AGENTS:
        return {"choices": [{"message": {"content": json.dumps({"error": f"Unknown agent: {agent_id}"})}}]}
    
    try:
        result = run_agent(agent_id, GREEN_AGENT_URL)
        result_json = json.dumps(result, cls=NumpyEncoder)
        return {"choices": [{"message": {"content": result_json}}]}
    except Exception as e:
        print(f"Error in {agent_id} execution: {e}")
        return {"choices": [{"message": {"content": json.dumps({"error": str(e)})}}]}

# =============================================================================
# Legacy endpoint (runs first agent by default)
# =============================================================================
@app.post("/v1/chat/completions")
async def start_assessment(request: Request):
    default_agent = list(AGENTS.keys())[0]
    try:
        result = run_agent(default_agent, GREEN_AGENT_URL)
        result_json = json.dumps(result, cls=NumpyEncoder)
        return {"choices": [{"message": {"content": result_json}}]}
    except Exception as e:
        print(f"Error in white agent execution: {e}")
        return {"choices": [{"message": {"content": json.dumps({"error": str(e)})}}]}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/status")
def status():
    return {"status": "ok"}

# =============================================================================
# AgentBeats controller endpoints - list ALL agents
# =============================================================================
@app.get("/agents")
def list_agents():
    public_url = os.getenv("PUBLIC_URL", "http://localhost:9001")
    result = {}
    for agent_id in AGENTS:
        result[agent_id] = {
            "id": agent_id,
            "name": AGENTS[agent_id]["name"],
            "url": f"{public_url}/agents/{agent_id}",
            "status": "ready",
            "ready": True,
            "agent_card": get_agent_card(agent_id)
        }
    return result

@app.get("/agents/{agent_id}")
def get_agent_status(agent_id: str):
    if agent_id not in AGENTS:
        return {"error": f"Unknown agent: {agent_id}"}
    
    public_url = os.getenv("PUBLIC_URL", "http://localhost:9001")
    return {
        "id": agent_id,
        "name": AGENTS[agent_id]["name"],
        "url": f"{public_url}/agents/{agent_id}",
        "status": "ready",
        "ready": True,
        "agent_card": get_agent_card(agent_id)
    }

# =============================================================================
# A2A Protocol agent cards
# =============================================================================
@app.get("/.well-known/agent.json")
def agent_card_a2a():
    """Returns first agent's card for legacy compatibility."""
    from fastapi.responses import JSONResponse
    default_agent = list(AGENTS.keys())[0]
    return JSONResponse(content=get_agent_card(default_agent), media_type="application/json")

@app.get("/agents/{agent_id}/.well-known/agent.json")
def agent_card_by_id(agent_id: str):
    from fastapi.responses import JSONResponse
    if agent_id not in AGENTS:
        return JSONResponse(content={"error": f"Unknown agent: {agent_id}"}, status_code=404)
    return JSONResponse(content=get_agent_card(agent_id), media_type="application/json")

# =============================================================================
# Reset endpoints
# =============================================================================
@app.post("/reset")
def reset_agent():
    return {"status": "ok", "message": "All agents reset successfully"}

@app.post("/agents/{agent_id}/reset")
def reset_agent_by_id(agent_id: str):
    return {"status": "ok", "agent_id": agent_id, "message": "Agent reset successfully"}
