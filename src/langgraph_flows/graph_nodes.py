"""Base node implementations for LangGraph voice agents."""

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from livekit.agents import Agent, ChatContext, get_job_context

from .graph_schema import AgentState, NodeMetadata
from utils.logger import get_logger

logger = get_logger("langgraph-nodes")


class BaseGraphNode(ABC):
    """Base class for LangGraph nodes with LiveKit integration."""

    def __init__(self, metadata: NodeMetadata, agent_session: Any = None):
        self.metadata = metadata
        self.agent_session = agent_session
        self._agent_instance: Optional[Agent] = None

    @abstractmethod
    async def execute(self, state: AgentState) -> AgentState:
        """Execute the node logic.

        Args:
            state: Current agent state

        Returns:
            Updated agent state
        """
        pass

    def __call__(self, state: AgentState) -> AgentState:
        """Synchronous wrapper for LangGraph compatibility."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import nest_asyncio

                nest_asyncio.apply()
                return loop.run_until_complete(self.execute(state))
            else:
                return asyncio.run(self.execute(state))
        except RuntimeError:
            return asyncio.run(self.execute(state))

    def _create_agent(self, state: AgentState) -> Agent:
        """Creates a LiveKit Agent instance for this node."""
        instructions = (
            f"{self.metadata.system_prompt}\n\n{self.metadata.prompt_template}"
        )

        chat_ctx = None
        if self.metadata.preserve_context and state.get("chat_context"):
            chat_ctx = ChatContext()

        agent = Agent(
            instructions=instructions,
            chat_ctx=chat_ctx,
        )

        return agent

    async def _generate_response(
        self, state: AgentState, instructions: Optional[str] = None
    ) -> str:
        """Generate agent response using LiveKit session.

        Args:
            state: Current state
            instructions: Optional override instructions

        Returns:
            Generated response text
        """
        if not self.agent_session:
            logger.warning("no_agent_session", node_id=self.metadata.id)
            return f"[{self.metadata.name}] Processing..."

        return f"Response from {self.metadata.name}"

    def _update_state(self, state: AgentState, **updates) -> AgentState:
        """Updates state immutably."""
        new_state = state.copy()
        new_state.update(updates)
        return new_state


class WelcomeGraphNode(BaseGraphNode):
    """Welcome/entry node."""

    async def execute(self, state: AgentState) -> AgentState:
        """Executes welcome logic."""
        logger.info("executing_welcome_node", node_id=self.metadata.id)

        node_history = state.get("node_history", [])
        node_history.append(self.metadata.id)

        if self.agent_session:
            # In actual implementation, this would generate via LiveKit
            response = await self._generate_response(
                state,
                instructions="Greet the user warmly and explain available options.",
            )
        else:
            response = f"Welcome! How can I help you today?"

        return self._update_state(
            state,
            current_node=self.metadata.id,
            previous_node=state.get("current_node"),
            node_history=node_history,
            agent_response=response,
            should_continue=True,
        )


class ConversationalGraphNode(BaseGraphNode):
    """Conversational node for extended interactions."""

    async def execute(self, state: AgentState) -> AgentState:
        """Execute conversational logic."""
        logger.info("executing_conversational_node", node_id=self.metadata.id)

        node_history = state.get("node_history", [])
        node_history.append(self.metadata.id)

        user_input = state.get("user_input", "")

        # Generate contextual response
        response = await self._generate_response(
            state, instructions=f"Respond to: {user_input}"
        )

        messages = state.get("messages", [])
        if user_input:
            messages.append({"role": "user", "content": user_input})
        messages.append({"role": "assistant", "content": response})

        return self._update_state(
            state,
            current_node=self.metadata.id,
            previous_node=state.get("current_node"),
            node_history=node_history,
            agent_response=response,
            messages=messages,
            should_continue=True,
        )


class ToolGraphNode(BaseGraphNode):
    """Node for tool execution."""

    async def execute(self, state: AgentState) -> AgentState:
        """Executes tool node logic."""
        logger.info(
            "executing_tool_node", node_id=self.metadata.id, tools=self.metadata.tools
        )

        node_history = state.get("node_history", [])
        node_history.append(self.metadata.id)

        tool_results = state.get("tool_results", {})

        pending_tools = state.get("pending_tools", [])
        for tool_name in pending_tools:
            if tool_name in self.metadata.tools:
                tool_results[tool_name] = {"status": "executed", "result": "success"}
                logger.info("tool_executed", tool_name=tool_name)

        return self._update_state(
            state,
            current_node=self.metadata.id,
            previous_node=state.get("current_node"),
            node_history=node_history,
            tool_results=tool_results,
            pending_tools=[],
            should_continue=True,
        )


class RouterGraphNode(BaseGraphNode):
    """Router node for decision making."""

    async def execute(self, state: AgentState) -> AgentState:
        """Executes router logic to determine next action."""
        logger.info("executing_router_node", node_id=self.metadata.id)

        node_history = state.get("node_history", [])
        node_history.append(self.metadata.id)

        user_input = state.get("user_input", "") or ""
        user_input_lower = user_input.lower() if user_input else ""

        next_action = "default"

        if user_input_lower and any(
            word in user_input_lower for word in ["billing", "payment", "invoice"]
        ):
            next_action = "billing"
        elif user_input_lower and any(
            word in user_input_lower
            for word in ["technical", "support", "help", "issue"]
        ):
            next_action = "technical"
        elif user_input_lower and any(
            word in user_input_lower for word in ["bye", "goodbye", "exit", "quit"]
        ):
            next_action = "hangup"

        logger.info("routing_decision", next_action=next_action, user_input=user_input)

        return self._update_state(
            state,
            current_node=self.metadata.id,
            previous_node=state.get("current_node"),
            node_history=node_history,
            next_action=next_action,
            should_continue=True,
        )


class HangupGraphNode(BaseGraphNode):
    """Hangup/termination node."""

    async def execute(self, state: AgentState) -> AgentState:
        """Execute hangup logic."""
        logger.info("executing_hangup_node", node_id=self.metadata.id)

        node_history = state.get("node_history", [])
        node_history.append(self.metadata.id)

        # Generate farewell
        response = await self._generate_response(
            state, instructions="Provide a warm, professional farewell."
        )

        # Mark as complete
        logger.info("session_ending", node_id=self.metadata.id)

        return self._update_state(
            state,
            current_node=self.metadata.id,
            previous_node=state.get("current_node"),
            node_history=node_history,
            agent_response=response,
            should_continue=False,
            is_complete=True,
        )


# Node type mapping
GRAPH_NODE_CLASSES = {
    "welcome": WelcomeGraphNode,
    "conversational": ConversationalGraphNode,
    "tool": ToolGraphNode,
    "router": RouterGraphNode,
    "hangup": HangupGraphNode,
}
