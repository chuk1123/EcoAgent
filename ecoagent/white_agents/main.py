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
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

GREEN_AGENT_URL = os.getenv("GREEN_AGENT_URL")
TARGET_AGENT_MODULE = "ecoagent.white_agents.react_agent"

@app.post("/v1/chat/completions")
async def start_assessment(request: Request):
    proxy = GreenAgentProxy(GREEN_AGENT_URL)
    
    try:
        agent_module = import_module(TARGET_AGENT_MODULE)
        importlib.reload(agent_module)
        
        result = agent_module.run(proxy)
        
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

# AgentBeats controller endpoint - returns dict of agent instances
@app.get("/agents")
def list_agents():
    agent_id = "white-agent-react"
    public_url = os.getenv("PUBLIC_URL", "http://localhost:9001")
    return {
        agent_id: {
            "id": agent_id,
            "name": "EcoAgent White Agent",
            "url": public_url,
            "status": "running",
            "agent_card": get_agent_card()
        }
    }

def get_agent_card():
    public_url = os.getenv("PUBLIC_URL", "http://localhost:9001")
    return {
        "name": "EcoAgent White Agent",
        "version": "1.0.0",
        "protocolVersion": "0.3.0",
        "description": "White Agent for EcoAgent (A2A). ReAct-based agent for resource-constrained housing forecasting.",
        "url": public_url + "/",
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
    from fastapi.responses import JSONResponse
    return JSONResponse(content=get_agent_card(), media_type="application/json")

# Legacy path for backwards compatibility
@app.get("/.well-known/agent-card.json")
def agent_card_legacy():
    from fastapi.responses import JSONResponse
    return JSONResponse(content=get_agent_card(), media_type="application/json")