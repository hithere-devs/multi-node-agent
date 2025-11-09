"""Node-based agent implementations using LiveKit Agent handoffs."""

import re
from typing import Any, Dict, Optional

from livekit.agents import (
    Agent,
    ChatContext,
    RunContext,
    function_tool,
    get_job_context,
)

from .schema import NodeConfig, NodeType, TransitionCondition, TransitionConditionType
from utils.logger import get_logger

logger = get_logger("node-agents")


class BaseNodeAgent(Agent):
    """Base class for all node agents with transition logic."""

    def __init__(
        self,
        node_config: NodeConfig,
        workflow_state: Dict[str, Any],
        chat_ctx: Optional[ChatContext] = None,
    ):
        """Initialize a node agent.

        Args:
            node_config: Configuration for this node
            workflow_state: Shared state across the workflow
            chat_ctx: Optional chat context to preserve conversation history
        """
        # Build instructions from prompt config
        instructions = (
            f"{node_config.prompt.system_prompt}\n\n{node_config.prompt.instructions}"
        )

        super().__init__(
            instructions=instructions,
            chat_ctx=chat_ctx,
        )

        self.node_config = node_config
        self.workflow_state = workflow_state
        self._transition_requested: Optional[str] = None

    async def on_enter(self) -> None:
        """Called when this agent becomes active."""
        logger.info(
            "node_entered",
            node_id=self.node_config.id,
            node_type=self.node_config.type.value,
        )

        # Track visited nodes
        if "visited_nodes" not in self.workflow_state:
            self.workflow_state["visited_nodes"] = []
        self.workflow_state["visited_nodes"].append(self.node_config.id)

        # Generate initial greeting/response
        await self._on_node_enter()

    async def _on_node_enter(self) -> None:
        """Override in subclasses for node-specific entry behavior."""
        await self.session.generate_reply()

    def _evaluate_transition(
        self, condition: TransitionCondition, context: Dict[str, Any]
    ) -> bool:
        """Evaluate if a transition condition is met.

        Args:
            condition: The transition condition to evaluate
            context: Current context including user input, tool results, etc.

        Returns:
            True if condition is met, False otherwise
        """
        if condition.type == TransitionConditionType.ALWAYS:
            return True

        elif condition.type == TransitionConditionType.REGEX:
            if "user_input" in context and condition.expression:
                return bool(
                    re.search(
                        condition.expression, context["user_input"], re.IGNORECASE
                    )
                )

        elif condition.type == TransitionConditionType.INTENT:
            # Intent matching would be handled by LLM tools
            # This is a placeholder for tool-based intent detection
            return context.get("detected_intent") == condition.intent_description

        elif condition.type == TransitionConditionType.TOOL_RESULT:
            tool_results = context.get("tool_results", {})
            if condition.tool_name in tool_results:
                result = tool_results[condition.tool_name]
                if condition.expected_value is not None:
                    return result == condition.expected_value
                return bool(result)

        elif condition.type == TransitionConditionType.EXPRESSION:
            # Evaluate Python expression with context
            if condition.expression:
                try:
                    return bool(
                        eval(condition.expression, {"__builtins__": {}}, context)
                    )
                except Exception as e:
                    logger.error("expression_evaluation_failed", error=str(e))
                    return False

        return False

    def _get_next_node_id(self, context: Dict[str, Any]) -> Optional[str]:
        """Determine the next node based on transitions.

        Args:
            context: Current context for evaluation

        Returns:
            Node ID to transition to, or None if no transition matches
        """
        # Sort by priority (highest first)
        sorted_transitions = sorted(
            self.node_config.transitions,
            key=lambda t: t.priority,
            reverse=True,
        )

        for transition in sorted_transitions:
            if self._evaluate_transition(transition.condition, context):
                logger.info(
                    "transition_matched",
                    from_node=self.node_config.id,
                    to_node=transition.target_node_id,
                    condition_type=transition.condition.type.value,
                )
                return transition.target_node_id

        return None

    def _create_transition_tool(self, node_id: str, tool_name: str, description: str):
        """Create a tool for explicit transitions."""

        @function_tool(name=tool_name, description=description)
        async def transition_tool(context: RunContext):
            """Tool to transition to a specific node."""
            self._transition_requested = node_id
            # Import here to avoid circular dependency
            from .orchestrator import get_node_agent

            next_agent = get_node_agent(
                node_id,
                self.workflow_state,
                chat_ctx=(
                    self.chat_ctx if self.node_config.preserve_chat_context else None
                ),
            )
            return next_agent, f"Transitioning to {node_id}"

        return transition_tool


class WelcomeNodeAgent(BaseNodeAgent):
    """Agent for welcome/entry nodes."""

    def __init__(
        self,
        node_config: NodeConfig,
        workflow_state: Dict[str, Any],
        chat_ctx: Optional[ChatContext] = None,
    ):
        super().__init__(node_config, workflow_state, chat_ctx)

        # Add transition tools based on configured transitions
        for transition in node_config.transitions:
            if transition.condition.type == TransitionConditionType.INTENT:
                tool_name = f"go_to_{transition.target_node_id}"
                description = (
                    transition.condition.intent_description
                    or f"Navigate to {transition.target_node_id}"
                )
                tool = self._create_transition_tool(
                    transition.target_node_id, tool_name, description
                )
                # Register the tool dynamically
                setattr(self, tool_name, tool)

    async def _on_node_enter(self) -> None:
        """Welcome node greeting."""
        await self.session.generate_reply(
            instructions="Greet the user warmly and present the available options clearly."
        )


class ConversationalNodeAgent(BaseNodeAgent):
    """Agent for conversational nodes with extended interactions."""

    async def _on_node_enter(self) -> None:
        """Start conversation."""
        await self.session.generate_reply(
            instructions=self.node_config.prompt.instructions
        )


class TransitionNodeAgent(BaseNodeAgent):
    """Agent for quick transition nodes (routing/decision points)."""

    async def _on_node_enter(self) -> None:
        """Immediately evaluate and transition."""
        context = {
            "workflow_state": self.workflow_state,
            "session_data": self.workflow_state.get("session_data", {}),
        }

        next_node = self._get_next_node_id(context)

        if next_node:
            # Import here to avoid circular dependency
            from .orchestrator import get_node_agent

            next_agent = get_node_agent(
                next_node,
                self.workflow_state,
                chat_ctx=(
                    self.chat_ctx if self.node_config.preserve_chat_context else None
                ),
            )

            # Quick acknowledgment before transition
            await self.session.generate_reply(
                instructions=f"Briefly acknowledge and transition to next step."
            )

            # Update session with new agent
            self.session.update_agent(next_agent)
        else:
            logger.warning("no_transition_found", node_id=self.node_config.id)
            await self.session.generate_reply(
                instructions="I'm not sure where to go next. Let me help you with something else."
            )


class ToolNodeAgent(BaseNodeAgent):
    """Agent for tool-focused nodes."""

    def __init__(
        self,
        node_config: NodeConfig,
        workflow_state: Dict[str, Any],
        chat_ctx: Optional[ChatContext] = None,
    ):
        super().__init__(node_config, workflow_state, chat_ctx)

        # Add tools specified in config
        # Tools would be registered from a tool registry
        # For now, we'll add transition tools
        for transition in node_config.transitions:
            if transition.condition.type == TransitionConditionType.TOOL_RESULT:
                # Create tool that triggers transition on specific result
                tool_name = transition.condition.tool_name
                if tool_name:
                    # Tool would be loaded from registry here
                    pass

    async def _on_node_enter(self) -> None:
        """Introduce available tools."""
        await self.session.generate_reply(
            instructions="Explain available actions and ask what the user needs."
        )


class HangupNodeAgent(BaseNodeAgent):
    """Agent for conversation termination."""

    async def _on_node_enter(self) -> None:
        """Farewell message and end session."""
        await self.session.generate_reply(
            instructions="Provide a warm, professional farewell message."
        )

        # Mark workflow as complete
        self.workflow_state["is_complete"] = True

        # End the call/session
        job_ctx = get_job_context()
        # Give time for the message to be delivered
        import asyncio

        await asyncio.sleep(2)

        # Disconnect
        await job_ctx.room.disconnect()


# Node type to agent class mapping
NODE_AGENT_CLASSES = {
    NodeType.WELCOME: WelcomeNodeAgent,
    NodeType.CONVERSATIONAL: ConversationalNodeAgent,
    NodeType.TRANSITION: TransitionNodeAgent,
    NodeType.TOOL: ToolNodeAgent,
    NodeType.HANGUP: HangupNodeAgent,
}
