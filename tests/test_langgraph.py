"""Tests for LangGraph-based orchestration."""

import pytest
from pathlib import Path

from langgraph_flows import (
    AgentState,
    NodeMetadata,
    GraphConfig,
    ConditionalEdge,
    VoiceGraphBuilder,
    load_graph_config_from_json,
    GraphExecutor,
    WelcomeGraphNode,
    ConversationalGraphNode,
    RouterGraphNode,
    HangupGraphNode,
)


@pytest.fixture
def simple_graph_config():
    """Create a simple graph configuration for testing."""
    return GraphConfig(
        id="test-graph",
        name="Test Graph",
        version="1.0",
        description="Test graph",
        entry_node="welcome",
        end_node="__end__",
        nodes=[
            NodeMetadata(
                id="welcome",
                name="Welcome",
                description="Welcome node",
                node_type="welcome",
                prompt_template="Welcome!",
                system_prompt="You are helpful.",
            ),
            NodeMetadata(
                id="router",
                name="Router",
                description="Router node",
                node_type="router",
                prompt_template="Routing...",
                system_prompt="Route the user.",
            ),
            NodeMetadata(
                id="conversation",
                name="Conversation",
                description="Chat node",
                node_type="conversational",
                prompt_template="Let's chat.",
                system_prompt="You are chatty.",
                preserve_context=True,
            ),
            NodeMetadata(
                id="hangup",
                name="Hangup",
                description="End node",
                node_type="hangup",
                prompt_template="Goodbye!",
                system_prompt="Say bye.",
            ),
        ],
        edges=[
            ("welcome", "router"),
            ("conversation", "hangup"),
        ],
        conditional_edges=[
            ConditionalEdge(
                source_node="router",
                condition_name="route_decision",
                target_nodes={
                    "chat": "conversation",
                    "end": "hangup",
                },
                default_node="conversation",
            )
        ],
    )


class TestGraphSchema:
    """Test graph schema definitions."""

    def test_agent_state_creation(self):
        """Test creating agent state."""
        state: AgentState = {
            "current_node": "welcome",
            "previous_node": None,
            "node_history": [],
            "messages": [],
            "user_input": "Hello",
            "agent_response": None,
            "session_id": "test-session",
            "user_id": None,
            "session_data": {},
            "next_action": None,
            "should_continue": True,
            "is_complete": False,
            "error": None,
            "tool_results": {},
            "pending_tools": [],
            "chat_context": [],
            "preserve_context": False,
        }

        assert state["current_node"] == "welcome"
        assert state["should_continue"] is True
        assert len(state["node_history"]) == 0

    def test_node_metadata_creation(self):
        """Test node metadata."""
        metadata = NodeMetadata(
            id="test-node",
            name="Test Node",
            description="A test node",
            node_type="conversational",
            prompt_template="Test prompt",
            system_prompt="System prompt",
            tools=["tool1", "tool2"],
        )

        assert metadata.id == "test-node"
        assert metadata.node_type == "conversational"
        assert len(metadata.tools) == 2


class TestGraphBuilder:
    """Test graph builder functionality."""

    def test_builder_initialization(self, simple_graph_config):
        """Test builder initializes correctly."""
        builder = VoiceGraphBuilder(simple_graph_config)

        assert builder.config.id == "test-graph"
        assert len(builder.nodes) == 0  # Not built yet

    def test_graph_build(self, simple_graph_config):
        """Test building the graph."""
        builder = VoiceGraphBuilder(simple_graph_config)
        graph = builder.build()

        assert graph is not None
        assert len(builder.nodes) == 4

    def test_node_creation(self, simple_graph_config):
        """Test node instance creation."""
        builder = VoiceGraphBuilder(simple_graph_config)

        welcome_meta = simple_graph_config.nodes[0]
        welcome_node = builder._create_node_instance(welcome_meta)

        assert isinstance(welcome_node, WelcomeGraphNode)
        assert welcome_node.metadata.id == "welcome"


class TestGraphNodes:
    """Test individual graph node types."""

    @pytest.mark.asyncio
    async def test_welcome_node_execution(self):
        """Test welcome node execution."""
        metadata = NodeMetadata(
            id="welcome",
            name="Welcome",
            description="Test welcome",
            node_type="welcome",
            prompt_template="Welcome!",
            system_prompt="Be welcoming.",
        )

        node = WelcomeGraphNode(metadata)

        initial_state: AgentState = {
            "current_node": "start",
            "previous_node": None,
            "node_history": [],
            "messages": [],
            "user_input": None,
            "agent_response": None,
            "session_id": "test",
            "user_id": None,
            "session_data": {},
            "next_action": None,
            "should_continue": True,
            "is_complete": False,
            "error": None,
            "tool_results": {},
            "pending_tools": [],
            "chat_context": [],
            "preserve_context": False,
        }

        result = await node.execute(initial_state)

        assert result["current_node"] == "welcome"
        assert "welcome" in result["node_history"]
        assert result["should_continue"] is True

    @pytest.mark.asyncio
    async def test_router_node_execution(self):
        """Test router node execution."""
        metadata = NodeMetadata(
            id="router",
            name="Router",
            description="Test router",
            node_type="router",
            prompt_template="Routing",
            system_prompt="Route users.",
        )

        node = RouterGraphNode(metadata)

        state: AgentState = {
            "current_node": "welcome",
            "previous_node": None,
            "node_history": ["welcome"],
            "messages": [],
            "user_input": "I need technical support",
            "agent_response": None,
            "session_id": "test",
            "user_id": None,
            "session_data": {},
            "next_action": None,
            "should_continue": True,
            "is_complete": False,
            "error": None,
            "tool_results": {},
            "pending_tools": [],
            "chat_context": [],
            "preserve_context": False,
        }

        result = await node.execute(state)

        assert result["current_node"] == "router"
        assert result["next_action"] == "technical"

    @pytest.mark.asyncio
    async def test_hangup_node_execution(self):
        """Test hangup node execution."""
        metadata = NodeMetadata(
            id="hangup",
            name="Hangup",
            description="Test hangup",
            node_type="hangup",
            prompt_template="Goodbye!",
            system_prompt="Say goodbye.",
        )

        node = HangupGraphNode(metadata)

        state: AgentState = {
            "current_node": "conversation",
            "previous_node": "welcome",
            "node_history": ["welcome", "conversation"],
            "messages": [],
            "user_input": None,
            "agent_response": None,
            "session_id": "test",
            "user_id": None,
            "session_data": {},
            "next_action": None,
            "should_continue": True,
            "is_complete": False,
            "error": None,
            "tool_results": {},
            "pending_tools": [],
            "chat_context": [],
            "preserve_context": False,
        }

        result = await node.execute(state)

        assert result["is_complete"] is True
        assert result["should_continue"] is False
        assert "hangup" in result["node_history"]


class TestGraphExecution:
    """Test graph execution."""

    @pytest.mark.asyncio
    async def test_executor_initialization(self):
        """Test executor initialization with config file."""
        config_path = (
            Path(__file__).parent.parent
            / "configs"
            / "langgraph"
            / "customer_support.json"
        )

        if config_path.exists():
            executor = GraphExecutor(str(config_path))

            assert executor.config.id == "customer-support-graph"
            assert executor.graph is not None

    def test_initial_state_creation(self, simple_graph_config):
        """Test creating initial state."""
        # Create a temporary config file
        import tempfile
        import json

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            # Convert config to dict (simplified)
            config_dict = {
                "id": simple_graph_config.id,
                "name": simple_graph_config.name,
                "version": simple_graph_config.version,
                "description": simple_graph_config.description,
                "entry_node": simple_graph_config.entry_node,
                "end_node": simple_graph_config.end_node,
                "nodes": [
                    {
                        "id": n.id,
                        "name": n.name,
                        "description": n.description,
                        "type": n.node_type,
                        "system_prompt": n.system_prompt,
                        "prompt_template": n.prompt_template,
                    }
                    for n in simple_graph_config.nodes
                ],
                "edges": list(simple_graph_config.edges),
                "conditional_edges": [],
            }
            json.dump(config_dict, f)
            temp_path = f.name

        try:
            executor = GraphExecutor(temp_path)
            initial_state = executor.create_initial_state(
                session_id="test-123", user_id="user-456"
            )

            assert initial_state["session_id"] == "test-123"
            assert initial_state["user_id"] == "user-456"
            assert initial_state["current_node"] == "welcome"
            assert initial_state["should_continue"] is True
        finally:
            import os

            os.unlink(temp_path)


class TestConfigLoading:
    """Test configuration loading."""

    def test_load_customer_support_graph(self):
        """Test loading customer support graph config."""
        config_path = (
            Path(__file__).parent.parent
            / "configs"
            / "langgraph"
            / "customer_support.json"
        )

        if config_path.exists():
            config = load_graph_config_from_json(str(config_path))

            assert config.id == "customer-support-graph"
            assert config.entry_node == "welcome"
            assert len(config.nodes) > 0
            assert len(config.conditional_edges) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
