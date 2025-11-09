"""Schema definitions for LangGraph-based voice agent orchestration."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Callable
from typing_extensions import TypedDict


class NodeStatus(str, Enum):
    """Node execution status."""

    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    ERROR = "error"


class AgentState(TypedDict, total=False):
    """State flowing through the LangGraph."""

    current_node: str
    previous_node: Optional[str]
    node_history: List[str]

    messages: List[Dict[str, Any]]
    user_input: Optional[str]
    agent_response: Optional[str]

    session_id: str
    user_id: Optional[str]
    session_data: Dict[str, Any]

    next_action: Optional[str]
    should_continue: bool
    is_complete: bool
    error: Optional[str]

    tool_results: Dict[str, Any]
    pending_tools: List[str]

    chat_context: List[Dict[str, str]]
    preserve_context: bool


@dataclass
class NodeMetadata:
    """Configuration metadata for a graph node."""

    id: str
    name: str
    description: str
    node_type: str
    prompt_template: str
    system_prompt: str
    max_tokens: int = 150
    temperature: float = 0.7
    tools: List[str] = field(default_factory=list)
    preserve_context: bool = False


@dataclass
class ConditionalEdge:
    """Defines a conditional edge in the graph."""

    source_node: str
    condition_name: str
    target_nodes: Dict[str, str]  # condition result -> target node
    default_node: Optional[str] = None


@dataclass
class GraphConfig:
    """Configuration for the entire LangGraph."""

    id: str
    name: str
    version: str
    description: str
    entry_node: str
    end_node: str
    nodes: List[NodeMetadata]
    edges: List[tuple[str, str]]  # (source, target) for direct edges
    conditional_edges: List[ConditionalEdge]
    metadata: Dict[str, Any] = field(default_factory=dict)


# Type for node functions
NodeFunction = Callable[[AgentState], AgentState]
ConditionalFunction = Callable[[AgentState], str]
