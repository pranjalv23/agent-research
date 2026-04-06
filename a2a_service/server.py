import logging

from agent_sdk.a2a.factory import create_a2a_app as _create

from .agent_card import RESEARCH_AGENT_CARD
from .executor import ResearchAgentExecutor

logger = logging.getLogger("agent_research.a2a_server")


def create_a2a_app():
    """Build the A2A Starlette application for the research agent."""
    app = _create(RESEARCH_AGENT_CARD, ResearchAgentExecutor, "agent_research")
    logger.info("A2A application created for Research Agent")
    return app
