"""Multi-nodal router for healthcare and other dynamic agent workflows."""

import re
from typing import Any, Dict, List, Optional, Tuple

from agents.config_models import (
    MultiNodeAgentConfig,
    NodeTransitionConfig,
    ConditionConfig,
    ConditionType,
)
from utils.logger import get_logger

logger = get_logger("multi-node-router")


class ConversationContext:
    """Maintains conversation history and state across node transitions."""

    def __init__(self):
        self.messages: List[Dict[str, str]] = []
        self.variables: Dict[str, Any] = {}
        self.visited_nodes: List[str] = []
        self.node_history: List[Dict[str, Any]] = []

    def add_message(self, role: str, content: str) -> None:
        """Add message to conversation history.

        Args:
            role: "user" or "assistant"
            content: Message content
        """
        self.messages.append({"role": role, "content": content})

    def add_user_message(self, content: str) -> None:
        self.add_message("user", content)

    def add_assistant_message(self, content: str) -> None:
        self.add_message("assistant", content)

    def get_conversation_history(self) -> List[Dict[str, str]]:
        return self.messages.copy()

    def set_variable(self, key: str, value: Any) -> None:
        self.variables[key] = value

    def get_variable(self, key: str, default: Any = None) -> Any:
        return self.variables.get(key, default)

    def record_node_visit(self, node_id: str, response: str) -> None:
        """Records node visit in history."""
        self.visited_nodes.append(node_id)
        self.node_history.append({"node_id": node_id, "response": response})

    def get_last_node(self) -> Optional[str]:
        return self.visited_nodes[-1] if self.visited_nodes else None

    def has_visited_node(self, node_id: str) -> bool:
        return node_id in self.visited_nodes

    def get_context_for_llm(self) -> Dict[str, Any]:
        """Returns context formatted for LLM consumption."""
        return {
            "conversation_history": self.get_conversation_history(),
            "variables": self.variables.copy(),
            "visited_nodes": self.visited_nodes.copy(),
        }


class MultiNodeRouter:
    """Routes conversations between nodes based on conditions and context."""

    def __init__(self, config: MultiNodeAgentConfig, llm_provider=None):
        """Initialize router with multi-nodal configuration.

        Args:
            config: Multi-nodal agent configuration
            llm_provider: Optional LLM provider for intent-based routing
        """
        self.config = config
        self.nodes_map = {node.id: node for node in config.nodes}
        self.current_node_id = config.entryNode
        self.context = ConversationContext()
        self.llm_provider = llm_provider

    def get_current_node(self):
        return self.nodes_map.get(self.current_node_id)

    def get_node(self, node_id: str):
        return self.nodes_map.get(node_id)

    def get_context(self) -> ConversationContext:
        return self.context

    def reset_context(self) -> None:
        self.context = ConversationContext()

    async def _evaluate_intent_condition(
        self, condition: ConditionConfig, user_input: str
    ) -> bool:
        """Use LLM to classify user intent for routing.

        Args:
            condition: Intent condition with description
            user_input: User's input text

        Returns:
            True if intent matches, False otherwise
        """
        if not self.llm_provider:
            logger.warning("llm_provider_not_set_for_intent_routing")
            return False

        if not condition.expression:
            logger.warning("intent_condition_missing_expression")
            return False

        try:
            # Build intent classification prompt
            intent_prompt = f"""Given the user input: "{user_input}"

Does this match the intent: "{condition.expression}"?

Answer only YES or NO."""

            result = await self.llm_provider.call(
                prompt=intent_prompt,
                system_prompt="You are an intent classifier. Answer only YES or NO.",
                max_tokens=10,
                temperature=0,
            )

            response_text = result.get("text", "").strip().lower()
            return "yes" in response_text

        except Exception as e:
            logger.error("intent_evaluation_error", error=str(e))
            return False

    def evaluate_condition(
        self, condition: ConditionConfig, context: Dict[str, Any]
    ) -> bool:
        """Evaluate a condition based on context.

        Args:
            condition: Condition to evaluate
            context: Context dict with variables and user input

        Returns:
            True if condition is met, False otherwise
        """
        try:
            if condition.type == ConditionType.ALWAYS:
                return True

            elif condition.type == ConditionType.REGEX:
                if not condition.expression:
                    return False
                user_input = context.get("user_input", "").lower()
                return bool(re.search(condition.expression, user_input, re.IGNORECASE))

            elif condition.type == ConditionType.NOT:
                if not condition.conditions or len(condition.conditions) == 0:
                    return False
                return not self.evaluate_condition(condition.conditions[0], context)

            elif condition.type == ConditionType.TOOL_RESULT:
                result = context.get("tool_result")
                return result == condition.value

            return False

        except Exception as e:
            logger.error(
                "condition_evaluation_error", error=str(e), condition=condition.type
            )
            return False

    async def find_next_node_async(
        self, context: Dict[str, Any]
    ) -> Tuple[Optional[str], bool]:
        """Find next node with async support for intent-based routing.

        Args:
            context: Context containing user_input and other variables

        Returns:
            Tuple of (next_node_id, is_valid_transition)
        """
        current_node = self.get_current_node()
        if not current_node:
            logger.error("current_node_not_found", node_id=self.current_node_id)
            return None, False

        sorted_transitions = sorted(current_node.transitions, key=lambda t: -t.priority)

        for transition in sorted_transitions:
            if transition.condition.type == "intent":
                user_input = context.get("user_input", "")
                is_match = await self._evaluate_intent_condition(
                    transition.condition, user_input
                )
            else:
                is_match = self.evaluate_condition(transition.condition, context)

            if is_match:
                logger.info(
                    "transition_found",
                    from_node=self.current_node_id,
                    to_node=transition.targetNode,
                    priority=transition.priority,
                    condition_type=transition.condition.type,
                )
                return transition.targetNode, True

        fallback_node = self._find_fallback_node(context)
        if fallback_node:
            logger.info(
                "fallback_transition_used",
                from_node=self.current_node_id,
                to_node=fallback_node,
            )
            return fallback_node, False

        logger.debug(
            "no_transition_found",
            node_id=self.current_node_id,
            user_input=context.get("user_input", "")[:50],
        )
        return self.current_node_id, False

    def find_next_node(self, context: Dict[str, Any]) -> Tuple[Optional[str], bool]:
        """Find next node based on current node transitions and context (sync version).

        Note: This version only supports regex/variable conditions.
        Use find_next_node_async() for intent-based routing.

        Args:
            context: Context containing user_input and other variables

        Returns:
            Tuple of (next_node_id, is_valid_transition)
        """
        current_node = self.get_current_node()
        if not current_node:
            logger.error("current_node_not_found", node_id=self.current_node_id)
            return None, False

        sorted_transitions = sorted(current_node.transitions, key=lambda t: -t.priority)

        for transition in sorted_transitions:
            if self.evaluate_condition(transition.condition, context):
                logger.info(
                    "transition_found",
                    from_node=self.current_node_id,
                    to_node=transition.targetNode,
                    priority=transition.priority,
                )
                return transition.targetNode, True

        fallback_node = self._find_fallback_node(context)
        if fallback_node:
            logger.info(
                "fallback_transition_used",
                from_node=self.current_node_id,
                to_node=fallback_node,
            )
            return fallback_node, False

        logger.debug(
            "no_transition_found",
            node_id=self.current_node_id,
            user_input=context.get("user_input", "")[:50],
        )
        return self.current_node_id, False

    def _find_fallback_node(self, context: Dict[str, Any]) -> Optional[str]:
        """Find a fallback node when no valid transition exists.

        Gracefully attempts to find an alternative node that can handle the user input.

        Args:
            context: Execution context

        Returns:
            Fallback node ID or None
        """
        # Strategy 1: Try to find entry node as safe fallback
        if self.config.entryNode != self.current_node_id:
            logger.debug(
                "using_entry_node_as_fallback", current_node=self.current_node_id
            )
            return self.config.entryNode

        return None

    def move_to_node(self, node_id: str) -> bool:
        """Moves router to specific node."""
        if node_id not in self.nodes_map:
            logger.error("invalid_node", node_id=node_id)
            return False

        old_node = self.current_node_id
        self.current_node_id = node_id
        self.context.record_node_visit(node_id, "")
        logger.info("node_changed", from_node=old_node, to_node=node_id)
        return True

    def get_available_transitions(
        self, context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Returns all available transitions from current node."""
        current_node = self.get_current_node()
        if not current_node:
            return []

        available = []
        for transition in current_node.transitions:
            if self.evaluate_condition(transition.condition, context):
                available.append(
                    {
                        "from_node": self.current_node_id,
                        "to_node": transition.targetNode,
                        "priority": transition.priority,
                        "condition_type": transition.condition.type,
                        "description": transition.description,
                    }
                )

        return sorted(available, key=lambda t: -t["priority"])

    def get_node_chain(
        self, from_node: str, to_node: str, max_depth: int = 10
    ) -> Optional[List[str]]:
        """Get path from one node to another.

        Args:
            from_node: Start node ID
            to_node: End node ID
            max_depth: Maximum search depth

        Returns:
            List of node IDs in path, or None if no path exists
        """
        if from_node not in self.nodes_map or to_node not in self.nodes_map:
            return None

        from collections import deque

        queue = deque([(from_node, [from_node])])
        visited = {from_node}
        depth = 0

        while queue and depth < max_depth:
            current, path = queue.popleft()
            depth += 1

            if current == to_node:
                return path

            node = self.nodes_map.get(current)
            if not node:
                continue

            for transition in node.transitions:
                next_node = transition.targetNode
                if next_node not in visited:
                    visited.add(next_node)
                    queue.append((next_node, path + [next_node]))

        return None

    def get_global_tools(self) -> List[str]:
        """Get list of global tools available in all nodes.

        Returns:
            List of global tool names
        """
        return self.config.globalTools.copy()

    def get_node_tools(self, node_id: Optional[str] = None) -> List[str]:
        """Get tools available in a node.

        Args:
            node_id: Node ID (uses current node if not provided)

        Returns:
            List of tool names (includes global tools)
        """
        node_id = node_id or self.current_node_id
        node = self.get_node(node_id)
        if not node:
            return self.get_global_tools()

        tools = set(node.tools or [])
        tools.update(self.get_global_tools())
        return list(tools)

    def build_system_prompt_with_paths(self) -> str:
        """Builds system prompt including available conversational paths."""
        current_node = self.get_current_node()
        if not current_node:
            return ""

        paths_info = "Available conversational paths from current state:\n"

        for i, transition in enumerate(current_node.transitions, 1):
            target_node = self.get_node(transition.targetNode)
            if target_node:
                paths_info += f"{i}. {transition.description or target_node.name}: "
                paths_info += f"Condition: {transition.condition.type}\n"

        return paths_info
