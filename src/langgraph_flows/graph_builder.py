"""LangGraph builder for voice agent orchestration."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from .graph_schema import (
    AgentState,
    NodeMetadata,
    GraphConfig,
    ConditionalEdge,
)
from .graph_nodes import GRAPH_NODE_CLASSES, BaseGraphNode
from utils.logger import get_logger

logger = get_logger("langgraph-builder")


class VoiceGraphBuilder:
    """Builds LangGraph state machines for voice agents."""

    def __init__(self, config: GraphConfig, agent_session: Any = None):
        self.config = config
        self.agent_session = agent_session
        self.nodes: Dict[str, BaseGraphNode] = {}

        # Create memory saver for checkpointing
        self.memory = MemorySaver()

        logger.info(
            "graph_builder_initialized",
            graph_id=config.id,
            node_count=len(config.nodes),
        )

    def _create_node_instance(self, metadata: NodeMetadata) -> BaseGraphNode:
        """Create a node instance from metadata.

        Args:
            metadata: Node metadata

        Returns:
            Node instance
        """
        node_class = GRAPH_NODE_CLASSES.get(metadata.node_type)

        if not node_class:
            raise ValueError(f"Unknown node type: {metadata.node_type}")

        return node_class(metadata, self.agent_session)

    def _create_routing_function(self, edge: ConditionalEdge):
        """Create a routing function for conditional edges.

        Args:
            edge: Conditional edge configuration

        Returns:
            Routing function
        """

        def route(state: AgentState) -> str:
            """Routes based on next_action in state."""
            next_action = state.get("next_action", "default")

            target = edge.target_nodes.get(next_action)

            if target:
                logger.info(
                    "routing",
                    from_node=edge.source_node,
                    to_node=target,
                    action=next_action,
                )
                return target

            if edge.default_node:
                logger.info(
                    "routing_default",
                    from_node=edge.source_node,
                    to_node=edge.default_node,
                    action=next_action,
                )
                return edge.default_node

            logger.warning(
                "no_route_found",
                from_node=edge.source_node,
                action=next_action,
            )
            return END

        return route

    def build(self) -> StateGraph:
        """Build the LangGraph state machine.

        Returns:
            Compiled state graph
        """
        # Create the graph
        graph = StateGraph(AgentState)

        for node_metadata in self.config.nodes:
            node_instance = self._create_node_instance(node_metadata)
            self.nodes[node_metadata.id] = node_instance

            # Add to graph
            graph.add_node(node_metadata.id, node_instance)
            logger.info(
                "node_added",
                node_id=node_metadata.id,
                node_type=node_metadata.node_type,
            )

        # Set entry point
        graph.set_entry_point(self.config.entry_node)
        logger.info("entry_point_set", entry_node=self.config.entry_node)

        # Add direct edges
        for source, target in self.config.edges:
            if target == self.config.end_node:
                graph.add_edge(source, END)
                logger.info("edge_added_to_end", source=source)
            else:
                graph.add_edge(source, target)
                logger.info("edge_added", source=source, target=target)

        # Add conditional edges
        for cond_edge in self.config.conditional_edges:
            routing_fn = self._create_routing_function(cond_edge)

            # Get possible target nodes
            targets = list(cond_edge.target_nodes.values())
            if cond_edge.default_node:
                targets.append(cond_edge.default_node)

            graph.add_conditional_edges(
                cond_edge.source_node,
                routing_fn,
                targets,
            )
            logger.info(
                "conditional_edge_added",
                source=cond_edge.source_node,
                targets=targets,
            )

        compiled_graph = graph.compile(checkpointer=self.memory)

        logger.info("graph_compiled", graph_id=self.config.id)

        return compiled_graph

    def visualize(self, output_path: Optional[str] = None) -> str:
        """Generate a visualization of the graph.

        Args:
            output_path: Optional path to save visualization

        Returns:
            Mermaid diagram as string
        """
        graph = self.build()

        try:
            mermaid = graph.get_graph().draw_mermaid()

            if output_path:
                Path(output_path).write_text(mermaid)
                logger.info("graph_visualization_saved", path=output_path)

            return mermaid
        except Exception as e:
            logger.error("visualization_failed", error=str(e))
            return ""


def load_graph_config_from_json(json_path: str) -> GraphConfig:
    """Load graph configuration from JSON file.

    Args:
        json_path: Path to JSON configuration

    Returns:
        Parsed graph configuration
    """
    path = Path(json_path)

    if not path.exists():
        raise FileNotFoundError(f"Graph config not found: {json_path}")

    with open(path, "r") as f:
        data = json.load(f)

    nodes = []
    for node_data in data.get("nodes", []):
        node = NodeMetadata(
            id=node_data.get("id"),
            name=node_data.get("name"),
            description=node_data.get("description", ""),
            node_type=node_data.get("type"),
            prompt_template=node_data.get("prompt_template", ""),
            system_prompt=node_data.get("system_prompt", ""),
            max_tokens=node_data.get("max_tokens", 150),
            temperature=node_data.get("temperature", 0.7),
            tools=node_data.get("tools", []),
            preserve_context=node_data.get("preserve_context", False),
        )
        nodes.append(node)

    edges = []
    for edge_data in data.get("edges", []):
        if isinstance(edge_data, list) and len(edge_data) == 2:
            edges.append((edge_data[0], edge_data[1]))
        elif isinstance(edge_data, dict):
            edges.append((edge_data.get("source"), edge_data.get("target")))

    conditional_edges = []
    for cond_data in data.get("conditional_edges", []):
        cond_edge = ConditionalEdge(
            source_node=cond_data.get("source"),
            condition_name=cond_data.get("condition"),
            target_nodes=cond_data.get("targets", {}),
            default_node=cond_data.get("default"),
        )
        conditional_edges.append(cond_edge)

    config = GraphConfig(
        id=data.get("id"),
        name=data.get("name"),
        version=data.get("version", "1.0"),
        description=data.get("description", ""),
        entry_node=data.get("entry_node"),
        end_node=data.get("end_node", "__end__"),
        nodes=nodes,
        edges=edges,
        conditional_edges=conditional_edges,
        metadata=data.get("metadata", {}),
    )

    logger.info(
        "graph_config_loaded",
        graph_id=config.id,
        node_count=len(config.nodes),
    )

    return config


def create_graph_from_config(
    config_path: str,
    agent_session: Any = None,
) -> StateGraph:
    """Create a compiled LangGraph from a JSON configuration.

    Args:
        config_path: Path to graph configuration JSON
        agent_session: Optional LiveKit AgentSession

    Returns:
        Compiled state graph
    """
    config = load_graph_config_from_json(config_path)
    builder = VoiceGraphBuilder(config, agent_session)
    return builder.build()
