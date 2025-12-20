"""Green Agent - A2A Protocol compliant implementation using official SDK."""

import json
import logging
import os

import uvicorn
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from a2a.utils import new_agent_text_message

from ecoagent.green_agent.green_agent import GreenAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [GREEN AGENT] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def prepare_agent_card(url: str) -> AgentCard:
    """Create the agent card for the green agent."""
    skill = AgentSkill(
        id="housing_evaluation",
        name="Housing Price Forecasting Evaluation",
        description="Evaluates white agents on resource-constrained housing price forecasting tasks.",
        tags=["green agent", "evaluation", "housing", "forecasting"],
        examples=[
            '{"action": "describe"}',
            '{"action": "request_dataset", "args": {"ds_id": "target", "split": "train"}}',
            '{"action": "evaluate_predictions", "args": {"y_pred": [100, 200, 300]}}',
        ],
    )
    return AgentCard(
        name="EcoAgent Green Agent",
        description="Green Agent evaluator for EcoAgent (A2A). Evaluates white agents on resource-constrained housing forecasting tasks.",
        url=url,
        version="1.0.0",
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        capabilities=AgentCapabilities(),
        skills=[skill],
    )


class GreenAgentExecutor(AgentExecutor):
    """Executor that handles incoming A2A messages for the green agent."""

    def __init__(self):
        self.green_agent = GreenAgent("ecoagent/contexts/housing")

    def process_command(self, command_text: str) -> str:
        """Process a command and return the result as a string."""
        try:
            command = json.loads(command_text)
        except json.JSONDecodeError:
            logger.warning(
                f"<<< RECEIVED INVALID JSON from WHITE AGENT: {command_text[:200]}"
            )
            return json.dumps({"error": "Invalid JSON format"})

        action = command.get("action")
        args = command.get("args", {})

        logger.info(f"<<< RECEIVED FROM WHITE AGENT: action={action}, args={args}")

        if action == "describe":
            info = self.green_agent.describe()
            result = json.dumps(info)
            logger.info(
                f">>> SENDING TO WHITE AGENT: describe info - context={info.get('context')}, target={info.get('target')}"
            )
            return result

        elif action == "request_dataset":
            ds_id = args.get("ds_id")
            split = args.get("split", "train")
            try:
                df = self.green_agent.request_dataset(ds_id, split)
                result = df.to_json(orient="split")
                logger.info(
                    f">>> SENDING TO WHITE AGENT: dataset '{ds_id}' ({split}) - {len(df)} rows, columns={list(df.columns)}"
                )
                return result
            except Exception as e:
                logger.error(f">>> SENDING TO WHITE AGENT: error - {str(e)}")
                return json.dumps({"error": str(e)})

        elif action == "evaluate_predictions":
            y_pred = args.get("y_pred", [])
            result = self.green_agent.evaluate_predictions(y_pred)
            logger.info(
                f">>> SENDING TO WHITE AGENT: evaluation - rmse={result.get('rmse')}, final_score={result.get('final_score')}, datasets_used={result.get('datasets_used')}"
            )
            return json.dumps(result)

        elif action == "reset":
            self.green_agent.reset()
            return json.dumps({"status": "ok", "message": "Agent reset successfully"})

        logger.warning(f">>> SENDING TO WHITE AGENT: error - Unknown action '{action}'")
        return json.dumps({"error": f"Unknown action: {action}"})

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Handle incoming A2A message."""
        user_input = context.get_user_input()
        logger.info(f"Received message: {user_input[:200]}...")

        # Process the command
        result_text = self.process_command(user_input)

        # Send response using proper A2A format
        await event_queue.enqueue_event(
            new_agent_text_message(result_text, context_id=context.context_id)
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Handle cancellation request."""
        raise NotImplementedError("Cancel not implemented")


def create_app():
    """Create and configure the A2A application."""
    # Get the agent URL from environment
    agent_url = os.getenv("AGENT_URL", os.getenv("PUBLIC_URL", "http://localhost:9000"))
    
    # Ensure URL ends with /
    if not agent_url.endswith("/"):
        agent_url = agent_url + "/"

    logger.info(f"Creating green agent with URL: {agent_url}")

    # Create agent card
    agent_card = prepare_agent_card(agent_url)

    # Create request handler with our executor
    request_handler = DefaultRequestHandler(
        agent_executor=GreenAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )

    # Create A2A application
    a2a_app = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )

    return a2a_app.build()


# Create the app instance for uvicorn
# Note: Environment variables should be set before importing this module
app = create_app()


def start_green_agent(host: str = "0.0.0.0", port: int = 9000):
    """Start the green agent server."""
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("AGENT_PORT", os.getenv("PORT", "9000")))
    start_green_agent(host=host, port=port)
