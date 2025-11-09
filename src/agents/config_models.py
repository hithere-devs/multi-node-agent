"""Pydantic models for multi-prompt agent configuration."""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ConditionType(str, Enum):
    """Types of conditions that can trigger state transitions."""

    REGEX = "regex"
    VARIABLE = "variable"
    TOOL_RESULT = "toolResult"
    ALWAYS = "always"
    AND = "and"
    OR = "or"
    NOT = "not"
    INTENT = "intent"


class NodeType(str, Enum):
    """Node types in multi-nodal workflows."""

    ENTRY = "entry"
    CONVERSATION = "conversation"
    SERVICE = "service"
    TOOL = "tool"
    DECISION = "decision"
    TRANSITION = "transition"
    EXIT = "exit"


class ConditionConfig(BaseModel):
    """Defines when and how a transition should occur."""

    type: ConditionType
    expression: Optional[str] = None
    variable: Optional[str] = None
    value: Optional[Any] = None
    conditions: Optional[List["ConditionConfig"]] = None


ConditionConfig.model_rebuild()


class TransitionConfig(BaseModel):
    """Defines transition between states with conditions and priority."""

    condition: ConditionConfig
    targetState: Optional[str] = None
    targetNode: Optional[str] = None
    priority: int = 0


class NodeTransitionConfig(BaseModel):
    """Transition configuration specific to multi-nodal systems."""

    condition: ConditionConfig
    targetNode: str
    priority: int = 0
    description: Optional[str] = None


class PromptConfig(BaseModel):
    """LLM prompt configuration with generation parameters."""

    template: str
    maxTokens: int = 100
    temperature: float = 0.7
    systemPrompt: Optional[str] = None


class ToolCallConfig(BaseModel):
    """Configuration for tools callable by the agent."""

    id: str
    type: str
    name: str
    description: str
    enabled: bool = True
    parameters: Optional[Dict[str, Any]] = None


class ToolConfig(BaseModel):
    """External tool configuration (HTTP endpoints, functions, etc)."""

    id: str
    type: str
    name: str
    description: str
    endpoint: Optional[str] = None
    method: Optional[str] = "POST"
    timeout: int = 30
    headers: Optional[Dict[str, str]] = None
    payloadTemplate: Optional[Dict[str, Any]] = None
    credentials: Optional[Dict[str, str]] = None


class StateConfig(BaseModel):
    """State machine state configuration."""

    id: str
    name: str
    prompt: PromptConfig
    tools: List[str] = Field(default_factory=list)
    transitions: List[TransitionConfig] = Field(default_factory=list)
    fallbackState: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class SessionDefaultsConfig(BaseModel):
    """Default provider settings for agent sessions."""

    sttModel: str = "assemblyai/universal-streaming"
    llmModel: str = "openai/gpt-4.1-mini"
    ttsModel: str = "cartesia/sonic-3"
    ttsVoice: Optional[str] = None
    language: str = "en"


class CustomerConfig(BaseModel):
    """Complete customer-specific agent configuration."""

    id: str
    name: str
    version: str = "1.0"
    entryState: str
    states: List[StateConfig]
    tools: Dict[str, ToolConfig] = Field(default_factory=dict)
    sessionDefaults: SessionDefaultsConfig = Field(
        default_factory=SessionDefaultsConfig
    )
    metadata: Optional[Dict[str, Any]] = None


class SessionConfig(BaseModel):
    """Runtime session configuration and metadata."""

    sessionId: str
    customerId: str
    customerConfig: CustomerConfig
    agentId: str
    participantId: str
    roomName: str
    ttlSeconds: int = 3600
    metadata: Optional[Dict[str, Any]] = None


class NodeType(str, Enum):
    """Node types in multi-nodal workflows."""

    ENTRY = "entry"
    SERVICE = "service"
    DECISION = "decision"
    EXIT = "exit"


class ToolDefinition(BaseModel):
    """Defines a tool's interface and requirements."""

    name: str
    description: str
    required: bool = False


class NodeContextRequirement(BaseModel):
    """Specifies required context data for a node."""

    field_name: str
    requirement_level: str
    description: Optional[str] = None


class MultiNodeConfig(BaseModel):
    """Single node configuration in multi-nodal workflow."""

    id: str
    type: NodeType
    name: str
    description: Optional[str] = None
    prompt: PromptConfig
    tools: List[str] = Field(default_factory=list)
    context: Optional[Dict[str, str]] = None
    transitions: List[NodeTransitionConfig] = Field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = None


class MultiNodeAgentConfig(BaseModel):
    """Complete multi-nodal agent configuration."""

    id: str
    name: str
    version: str = "2.0"
    description: Optional[str] = None
    agentInstructions: Optional[str] = None
    domainContext: Optional[str] = None
    entryNode: str
    nodes: List[MultiNodeConfig]
    globalTools: List[str] = Field(default_factory=list)
    validationRules: Optional[Dict[str, Any]] = None
    sessionDefaults: SessionDefaultsConfig = Field(
        default_factory=SessionDefaultsConfig
    )
    metadata: Optional[Dict[str, Any]] = None
