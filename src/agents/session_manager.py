"""Session management for multi-prompt agents."""

import time
import uuid
from typing import Any, Dict, Optional

from agents.config_models import CustomerConfig, SessionConfig
from agents.state_machine import SessionContext, StateMachineEngine
from utils.logger import get_logger

logger = get_logger(__name__)


class SessionManager:
    """Manages agent sessions and coordinates state machine execution."""

    def __init__(
        self,
        customer_config: CustomerConfig,
        state_machine: StateMachineEngine,
        room_name: str,
        participant_id: str,
        agent_id: str = "default-agent",
    ) -> None:
        self.customer_config = customer_config
        self.state_machine = state_machine
        self.room_name = room_name
        self.participant_id = participant_id
        self.agent_id = agent_id

        entry_state = state_machine.get_entry_state()
        self.session_context = SessionContext(
            session_id=str(uuid.uuid4()),
            customer_id=customer_config.id,
            agent_id=agent_id,
            current_state=entry_state.id,
        )

        self.created_at = time.time()
        self.last_activity = self.created_at
        self.is_active = True

        logger.info(
            "session_created",
            session_id=self.session_context.session_id,
            customer_id=customer_config.id,
            participant_id=participant_id,
            room_name=room_name,
        )

    async def process_user_input(self, user_text: str) -> str:
        """
        Process user input and get agent response.

        Args:
            user_text: User input text

        Returns:
            Agent response text
        """
        self.last_activity = time.time()

        try:
            response_text, next_state = await self.state_machine.process_user_input(
                self.session_context,
                user_text,
            )

            logger.info(
                "user_input_processed",
                session_id=self.session_context.session_id,
                response_length=len(response_text),
            )

            return response_text

        except Exception as e:
            logger.error(
                "process_input_failed",
                session_id=self.session_context.session_id,
                error=str(e),
            )
            raise

    def get_session_context(self) -> SessionContext:
        return self.session_context

    def get_session_info(self) -> Dict[str, Any]:
        """Returns session metadata and current state information."""
        elapsed = time.time() - self.created_at
        return {
            "session_id": self.session_context.session_id,
            "customer_id": self.session_context.customer_id,
            "agent_id": self.agent_id,
            "participant_id": self.participant_id,
            "room_name": self.room_name,
            "current_state": self.session_context.current_state,
            "is_active": self.is_active,
            "elapsed_seconds": elapsed,
            "history_length": len(self.session_context.history),
            "variables": self.session_context.variables,
        }

    def set_variable(self, key: str, value: Any) -> None:
        self.session_context.set_variable(key, value)

    def get_variable(self, key: str, default: Any = None) -> Any:
        return self.session_context.get_variable(key, default)

    async def close(self) -> None:
        """Closes the session and logs summary."""
        self.is_active = False
        elapsed = time.time() - self.created_at

        logger.info(
            "session_closed",
            session_id=self.session_context.session_id,
            elapsed_seconds=elapsed,
            history_length=len(self.session_context.history),
        )


class SessionRegistry:
    """Registry for managing multiple active sessions."""

    def __init__(self, ttl_seconds: int = 3600) -> None:
        """
        Initialize session registry.

        Args:
            ttl_seconds: Time-to-live for sessions
        """
        self.sessions: Dict[str, SessionManager] = {}
        self.ttl_seconds = ttl_seconds

    def register_session(self, session: SessionManager) -> str:
        """
        Register a new session.

        Args:
            session: Session manager instance

        Returns:
            Session ID
        """
        session_id = session.session_context.session_id
        self.sessions[session_id] = session
        logger.info("session_registered", session_id=session_id)
        return session_id

    def get_session(self, session_id: str) -> Optional[SessionManager]:
        return self.sessions.get(session_id)

    def remove_session(self, session_id: str) -> None:
        """Removes session from registry."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info("session_removed", session_id=session_id)

    async def cleanup_expired_sessions(self) -> None:
        """Cleans up expired sessions based on TTL."""
        current_time = time.time()
        expired_sessions = []

        for session_id, session in self.sessions.items():
            elapsed = current_time - session.created_at
            if elapsed > self.ttl_seconds:
                expired_sessions.append(session_id)

        for session_id in expired_sessions:
            session = self.sessions[session_id]
            await session.close()
            self.remove_session(session_id)
            logger.info("session_expired", session_id=session_id)

    async def close_all(self) -> None:
        """Closes all active sessions."""
        for session in list(self.sessions.values()):
            await session.close()
        self.sessions.clear()
