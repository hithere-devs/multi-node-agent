"""LiveKit integration for multi-prompt agents."""

from typing import Callable, Optional

from agents.config_models import CustomerConfig
from agents.prompt_renderer import PromptRenderer
from agents.session_manager import SessionManager, SessionRegistry
from agents.state_machine import StateMachineEngine
from agents.tool_invoker import ToolInvoker
from agents.llm_provider import LLMProvider, OpenAIProvider
from utils.logger import get_logger

logger = get_logger(__name__)


class LiveKitConnector:
    """
    Integrates the multi-prompt agent system with LiveKit.

    Bridges between LiveKit events (participant joined, audio transcription, etc.)
    and the state machine engine.
    """

    def __init__(
        self,
        customer_config: CustomerConfig,
        llm_provider: Optional[LLMProvider] = None,
        tool_registry: Optional[dict] = None,
    ) -> None:
        """
        Initialize LiveKit connector.

        Args:
            customer_config: Customer configuration
            llm_provider: LLM provider (defaults to OpenAI)
            tool_registry: Dictionary of tool handlers
        """
        self.customer_config = customer_config
        self.llm_provider = llm_provider or OpenAIProvider()
        self.tool_registry = tool_registry or {}

        # Initialize components
        self.renderer = PromptRenderer()
        self.tool_invoker = ToolInvoker(function_registry=self.tool_registry)
        self.state_machine = StateMachineEngine(
            config=customer_config,
            renderer=self.renderer,
            llm=self.llm_provider,
            tools=self.tool_invoker,
        )

        # Session management
        self.session_registry = SessionRegistry()

        logger.info(
            "livekit_connector_initialized",
            customer_id=customer_config.id,
        )

    async def create_session(
        self,
        room_name: str,
        participant_id: str,
        agent_id: str = "default-agent",
    ) -> SessionManager:
        """
        Create a new session for a participant.

        Args:
            room_name: LiveKit room name
            participant_id: Participant ID
            agent_id: Agent identifier

        Returns:
            SessionManager instance
        """
        session = SessionManager(
            customer_config=self.customer_config,
            state_machine=self.state_machine,
            room_name=room_name,
            participant_id=participant_id,
            agent_id=agent_id,
        )

        session_id = self.session_registry.register_session(session)

        logger.info(
            "session_created_for_participant",
            session_id=session_id,
            participant_id=participant_id,
            room_name=room_name,
        )

        return session

    def get_session(self, session_id: str) -> Optional[SessionManager]:
        """Get a session by ID."""
        return self.session_registry.get_session(session_id)

    async def close_session(self, session_id: str) -> None:
        """Close and cleanup a session."""
        session = self.session_registry.get_session(session_id)
        if session:
            await session.close()
            self.session_registry.remove_session(session_id)
            logger.info("session_closed_and_removed", session_id=session_id)

    async def cleanup(self) -> None:
        """Cleanup all sessions and resources."""
        await self.session_registry.close_all()
        await self.tool_invoker.close()
        logger.info("livekit_connector_cleanup_completed")

    def register_tool_handler(
        self,
        tool_id: str,
        handler: Callable,
    ) -> None:
        """
        Register a tool handler.

        Args:
            tool_id: Tool identifier
            handler: Async or sync handler function
        """
        self.tool_invoker.register_function(tool_id, handler)
        logger.info("tool_handler_registered", tool_id=tool_id)
