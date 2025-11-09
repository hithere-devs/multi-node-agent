"""Workflow orchestrator that manages node-based agents."""

import json
from pathlib import Path
from typing import Any, Dict, Optional

from livekit.agents import ChatContext

from .schema import (
    NodeConfig,
    NodeTransition,
    PromptConfig,
    TransitionCondition,
    TransitionConditionType,
    WorkflowConfig,
    NodeType,
)
from .node_agents import NODE_AGENT_CLASSES, BaseNodeAgent
from utils.logger import get_logger

logger = get_logger("workflow-orchestrator")


class WorkflowOrchestrator:
    """Orchestrates node-based workflow execution."""

    def __init__(self, workflow_config: WorkflowConfig):
        self.config = workflow_config
        self.nodes_by_id: Dict[str, NodeConfig] = {
            node.id: node for node in workflow_config.nodes
        }
        self.workflow_state: Dict[str, Any] = {
            "current_node_id": workflow_config.entry_node_id,
            "session_data": {},
            "visited_nodes": [],
            "transition_count": 0,
            "is_complete": False,
        }

        logger.info(
            "workflow_initialized",
            workflow_id=workflow_config.id,
            entry_node=workflow_config.entry_node_id,
            node_count=len(workflow_config.nodes),
        )

    def get_node_config(self, node_id: str) -> Optional[NodeConfig]:
        """Retrieves configuration for a specific node."""
        return self.nodes_by_id.get(node_id)

    def create_node_agent(
        self,
        node_id: str,
        chat_ctx: Optional[ChatContext] = None,
    ) -> BaseNodeAgent:
        """Create an agent instance for a specific node.

        Args:
            node_id: ID of the node to create agent for
            chat_ctx: Optional chat context to preserve conversation

        Returns:
            Agent instance for the node

        Raises:
            ValueError: If node not found or node type not supported
        """
        node_config = self.get_node_config(node_id)

        if not node_config:
            raise ValueError(f"Node not found: {node_id}")

        agent_class = NODE_AGENT_CLASSES.get(node_config.type)

        if not agent_class:
            raise ValueError(f"Unsupported node type: {node_config.type}")

        logger.info(
            "creating_node_agent",
            node_id=node_id,
            node_type=node_config.type.value,
            preserve_context=node_config.preserve_chat_context,
        )

        return agent_class(
            node_config=node_config,
            workflow_state=self.workflow_state,
            chat_ctx=chat_ctx,
        )

    def get_entry_agent(self) -> BaseNodeAgent:
        """Returns the initial entry agent for the workflow."""
        return self.create_node_agent(self.config.entry_node_id)


_orchestrator: Optional[WorkflowOrchestrator] = None


def initialize_orchestrator(workflow_config: WorkflowConfig) -> WorkflowOrchestrator:
    """Initializes the global workflow orchestrator."""
    global _orchestrator
    _orchestrator = WorkflowOrchestrator(workflow_config)
    return _orchestrator


def get_orchestrator() -> Optional[WorkflowOrchestrator]:
    """Returns the global workflow orchestrator."""
    return _orchestrator


def get_node_agent(
    node_id: str,
    workflow_state: Dict[str, Any],
    chat_ctx: Optional[ChatContext] = None,
) -> BaseNodeAgent:
    """Helper to get a node agent from the global orchestrator.

    Args:
        node_id: ID of the node
        workflow_state: Current workflow state
        chat_ctx: Optional chat context

    Returns:
        Agent instance for the node
    """
    orchestrator = get_orchestrator()
    if not orchestrator:
        raise RuntimeError("Workflow orchestrator not initialized")

    return orchestrator.create_node_agent(node_id, chat_ctx)


def load_workflow_from_json(json_path: str) -> WorkflowConfig:
    """Load workflow configuration from JSON file.

    Args:
        json_path: Path to JSON configuration file

    Returns:
        Parsed workflow configuration
    """
    path = Path(json_path)

    if not path.exists():
        raise FileNotFoundError(f"Workflow config not found: {json_path}")

    with open(path, "r") as f:
        data = json.load(f)

    # Parse the JSON into our schema
    nodes = []
    for node_data in data.get("nodes", []):
        # Parse prompt
        prompt_data = node_data.get("prompt", {})
        prompt = PromptConfig(
            system_prompt=prompt_data.get("system_prompt", ""),
            instructions=prompt_data.get("instructions", ""),
            max_tokens=prompt_data.get("max_tokens", 150),
            temperature=prompt_data.get("temperature", 0.7),
        )

        # Parse transitions
        transitions = []
        for trans_data in node_data.get("transitions", []):
            cond_data = trans_data.get("condition", {})
            condition = TransitionCondition(
                type=TransitionConditionType(cond_data.get("type", "always")),
                description=cond_data.get("description"),
                expression=cond_data.get("expression"),
                intent_description=cond_data.get("intent_description"),
                tool_name=cond_data.get("tool_name"),
                expected_value=cond_data.get("expected_value"),
            )

            transition = NodeTransition(
                target_node_id=trans_data.get("target_node_id"),
                condition=condition,
                priority=trans_data.get("priority", 5),
                metadata=trans_data.get("metadata", {}),
            )
            transitions.append(transition)

        node = NodeConfig(
            id=node_data.get("id"),
            type=NodeType(node_data.get("type")),
            name=node_data.get("name", ""),
            description=node_data.get("description", ""),
            prompt=prompt,
            tools=node_data.get("tools", []),
            transitions=transitions,
            metadata=node_data.get("metadata", {}),
            tts_voice=node_data.get("tts_voice"),
            preserve_chat_context=node_data.get("preserve_chat_context", False),
        )
        nodes.append(node)

    workflow = WorkflowConfig(
        id=data.get("id"),
        name=data.get("name"),
        version=data.get("version", "1.0"),
        description=data.get("description", ""),
        entry_node_id=data.get("entry_node_id"),
        nodes=nodes,
        global_tools=data.get("global_tools", []),
        metadata=data.get("metadata", {}),
    )

    logger.info(
        "workflow_loaded",
        workflow_id=workflow.id,
        node_count=len(workflow.nodes),
    )

    return workflow
