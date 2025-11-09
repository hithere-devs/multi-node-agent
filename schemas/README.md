# Schema Documentation Guide

Complete guide for building LiveKit voice agent configurations across all three
orchestration systems.

## Table of Contents

1. [Overview](#overview)
2. [Multi-Node Agent Schema](#multi-node-agent-schema)
3. [Livekit Workflow Agent Schema](#workflow-agent-schema)
4. [LangGraph Agent Schema](#langgraph-agent-schema)
5. [Comparison & Choosing a Schema](#comparison--choosing-a-schema)
6. [Best Practices](#best-practices)

---

## Overview

This project supports three distinct agent orchestration systems, each with its
own JSON configuration schema:

| System               | File                       | Best For                                               |
| -------------------- | -------------------------- | ------------------------------------------------------ |
| **Multi-Node Agent** | `configs/*.json`           | Complex state machines with variable-based transitions |
| **Workflow Agent**   | `configs/workflows/*.json` | Simple linear workflows with agent handoffs            |
| **LangGraph Agent**  | `configs/langgraph/*.json` | Advanced state management with conditional routing     |

All schemas are formally defined in the `schemas/` directory using JSON Schema
(Draft 7) for validation and IDE support.

---

## Multi-Node Agent Schema

**Schema File:** `schemas/multinode_schema.json` **Entry Point:** `src/agent.py`
**Example Configs:** `configs/ecommerce_multinode.json`,
`configs/healthcare_multinode.json`

### Core Concepts

The multi-node system uses a state machine architecture where:

- **Nodes** represent conversation states with specific prompts and behaviors
- **Transitions** define conditional routing between nodes
- **Conditions** use intent detection, regex, or variable checks
- **Context** is preserved throughout the conversation

### Required Fields

```json
{
  "id": "unique-agent-id",           // Lowercase, hyphenated identifier
  "name": "Human Readable Name",     // Display name
  "version": "1.0",                  // Semantic versioning
  "entryNode": "welcome",            // Starting node ID
  "nodes": [...]                     // Array of node objects
}
```

### Node Structure

Each node must have:

```json
{
  "id": "node_id",                   // Unique, snake_case identifier
  "type": "entry|conversational|tool|exit",
  "name": "Display Name",
  "description": "What this node does",
  "prompt": {
    "template": "Your prompt with {{variables}}",
    "systemPrompt": "LLM behavior instructions",
    "maxTokens": 150,
    "temperature": 0.7
  },
  "tools": ["tool1", "tool2"],       // Optional: node-specific tools
  "context": {                       // Optional: required context vars
    "user_input": "required",
    "order_id": "optional"
  },
  "transitions": [...]               // Array of transition rules
}
```

#### Node Types

1. **`entry`** - Initial greeting and routing node

   - First point of contact
   - Sets up conversation context
   - Routes to specialized nodes

2. **`conversational`** - Handles multi-turn dialogues

   - Main interaction nodes
   - Can access tools
   - Maintains conversation state

3. **`tool`** - Executes specific actions

   - Performs operations (database queries, API calls)
   - Minimal conversational interaction
   - Returns results and transitions

4. **`exit`** - Terminates conversation
   - Final goodbye message
   - Cleanup operations
   - Ends session

### Transition Structure

Transitions define routing logic:

```json
{
	"condition": {
		"type": "intent|regex|exact|variable|always",
		"expression": "condition details"
	},
	"targetNode": "destination_node_id",
	"priority": 10, // Higher = evaluated first
	"description": "When this fires"
}
```

#### Condition Types

1. **`intent`** - Natural language intent detection

   ```json
   {
   	"type": "intent",
   	"expression": "User wants to track their order status"
   }
   ```

2. **`regex`** - Pattern matching on user input

   ```json
   {
   	"type": "regex",
   	"expression": "(?i)(order|track|delivery|shipping)"
   }
   ```

3. **`exact`** - Exact string matching

   ```json
   {
   	"type": "exact",
   	"expression": "speak to supervisor"
   }
   ```

4. **`variable`** - Check context variables

   ```json
   {
   	"type": "variable",
   	"expression": "order_id != null && status == 'shipped'"
   }
   ```

5. **`always`** - Default/fallback transition
   ```json
   {
   	"type": "always",
   	"expression": ""
   }
   ```

### Complete Example

```json
{
	"id": "support-bot-v1",
	"name": "Customer Support Bot",
	"version": "1.0",
	"description": "Handles customer inquiries and routing",
	"agentInstructions": "You are a helpful customer support agent.",
	"domainContext": "E-commerce customer service",
	"entryNode": "welcome",
	"globalTools": ["hangup", "transfer"],
	"nodes": [
		{
			"id": "welcome",
			"type": "entry",
			"name": "Welcome Router",
			"description": "Greet and route customers",
			"prompt": {
				"template": "Hello! I can help with orders, returns, or account questions. What brings you here today?\n\nUser: {{user_input}}",
				"systemPrompt": "You are friendly and efficient. Route customers quickly.",
				"maxTokens": 100,
				"temperature": 0.7
			},
			"tools": ["get_customer_id"],
			"context": {
				"user_input": "required"
			},
			"transitions": [
				{
					"condition": {
						"type": "intent",
						"expression": "Customer needs help with an order"
					},
					"targetNode": "order_support",
					"priority": 10,
					"description": "Order-related inquiries"
				},
				{
					"condition": {
						"type": "regex",
						"expression": "(?i)(bye|goodbye|done)"
					},
					"targetNode": "goodbye",
					"priority": 9,
					"description": "Customer wants to end call"
				},
				{
					"condition": {
						"type": "always",
						"expression": ""
					},
					"targetNode": "general_support",
					"priority": 1,
					"description": "Default to general support"
				}
			]
		},
		{
			"id": "order_support",
			"type": "conversational",
			"name": "Order Support",
			"description": "Handle order inquiries",
			"prompt": {
				"template": "I'm here to help with your order. Can you provide your order number?\n\nUser: {{user_input}}",
				"systemPrompt": "You are an order specialist. Be thorough and helpful.",
				"maxTokens": 200,
				"temperature": 0.6
			},
			"tools": ["track_order", "modify_order"],
			"transitions": [
				{
					"condition": {
						"type": "intent",
						"expression": "Issue is resolved"
					},
					"targetNode": "goodbye",
					"priority": 10
				}
			]
		},
		{
			"id": "goodbye",
			"type": "exit",
			"name": "Goodbye",
			"description": "End conversation",
			"prompt": {
				"template": "Thank you for contacting us. Have a great day!",
				"systemPrompt": "End the call professionally.",
				"maxTokens": 50,
				"temperature": 0.7
			},
			"tools": [],
			"transitions": []
		}
	]
}
```

### Best Practices

1. **Priority Management**

   - Use priority 10+ for specific intent-based transitions
   - Use priority 5-9 for regex patterns
   - Use priority 1 for default/fallback transitions

2. **Prompt Design**

   - Keep templates concise (under 200 words)
   - Use `{{variables}}` for dynamic content
   - Include user input context when needed

3. **Node Organization**

   - Start with entry node for routing
   - Group related functionality in specialized nodes
   - Always provide exit paths

4. **Context Variables**
   - Mark critical variables as "required"
   - Use descriptive variable names
   - Validate variables in transitions

---

## Workflow Agent Schema

**Schema File:** `schemas/workflow_schema.json` **Entry Point:**
`src/workflow_agent.py` **Example Configs:**
`configs/workflows/customer_support.json`,
`configs/workflows/restaurant_ordering.json`

### Core Concepts

The workflow system uses agent handoffs:

- **Nodes** are independent agents with specialized roles
- **Handoffs** transfer control between agents
- **Context preservation** maintains conversation continuity
- **Simpler** than multi-node, more structured than raw agents

### Required Fields

```json
{
  "id": "workflow-id",
  "name": "Workflow Name",
  "version": "1.0",
  "entry_node_id": "welcome",        // Starting node
  "global_tools": [],                // Optional: shared tools
  "nodes": [...]
}
```

### Node Structure

```json
{
  "id": "node_id",
  "type": "welcome|conversational|transition|tool|hangup",
  "name": "Display Name",
  "description": "Node purpose",
  "prompt": {
    "system_prompt": "Agent personality and role",
    "instructions": "Specific task instructions",
    "max_tokens": 150,
    "temperature": 0.7
  },
  "tools": [],                       // Node-specific tools
  "transitions": [...],              // Handoff rules
  "preserve_chat_context": true      // Maintain history
}
```

#### Node Types

1. **`welcome`** - Entry point agent

   - Greets users
   - Understands initial intent
   - Routes to appropriate agent

2. **`conversational`** - Dialogue agents

   - Handle extended conversations
   - Specialized by domain (billing, tech support, etc.)
   - Can preserve or reset context

3. **`transition`** - Routing agents

   - Decision-making only
   - Minimal conversation
   - Quick handoffs

4. **`tool`** - Action agents

   - Execute operations
   - Return structured results
   - Auto-transition after completion

5. **`hangup`** - Termination agent
   - Final farewell
   - Session cleanup
   - Ends workflow

### Transition Structure

Workflow transitions are simpler than multi-node:

```json
{
	"target_node_id": "destination",
	"condition": {
		"type": "intent|regex|exact|always",
		"intent_description": "User wants billing help", // For intent type
		"pattern": "(?i)(billing|payment)", // For regex type
		"text": "exact match string" // For exact type
	},
	"priority": 10
}
```

### Complete Example

```json
{
	"id": "support-workflow",
	"name": "Customer Support Workflow",
	"version": "1.0",
	"description": "Multi-agent support workflow",
	"entry_node_id": "welcome",
	"global_tools": [],
	"nodes": [
		{
			"id": "welcome",
			"type": "welcome",
			"name": "Welcome Agent",
			"description": "Greet and route customers",
			"prompt": {
				"system_prompt": "You are a friendly customer service representative.",
				"instructions": "Greet warmly and identify their need. Route to billing, technical, or general support.",
				"max_tokens": 150,
				"temperature": 0.7
			},
			"tools": [],
			"transitions": [
				{
					"target_node_id": "billing_support",
					"condition": {
						"type": "intent",
						"intent_description": "Customer needs help with billing, payments, or invoices"
					},
					"priority": 10
				},
				{
					"target_node_id": "technical_support",
					"condition": {
						"type": "intent",
						"intent_description": "Customer needs technical help or troubleshooting"
					},
					"priority": 10
				},
				{
					"target_node_id": "hangup",
					"condition": {
						"type": "regex",
						"pattern": "(?i)(bye|goodbye|thanks)"
					},
					"priority": 9
				}
			],
			"preserve_chat_context": false
		},
		{
			"id": "billing_support",
			"type": "conversational",
			"name": "Billing Specialist",
			"description": "Handle billing inquiries",
			"prompt": {
				"system_prompt": "You are a billing specialist. Be thorough and empathetic.",
				"instructions": "Help with account questions, payments, and refunds. Ask clarifying questions as needed.",
				"max_tokens": 200,
				"temperature": 0.6
			},
			"tools": [],
			"transitions": [
				{
					"target_node_id": "hangup",
					"condition": {
						"type": "intent",
						"intent_description": "Customer is satisfied and ready to end"
					},
					"priority": 10
				},
				{
					"target_node_id": "welcome",
					"condition": {
						"type": "intent",
						"intent_description": "Customer needs a different department"
					},
					"priority": 8
				}
			],
			"preserve_chat_context": true
		},
		{
			"id": "hangup",
			"type": "hangup",
			"name": "Goodbye Agent",
			"description": "End the conversation",
			"prompt": {
				"system_prompt": "You end calls professionally.",
				"instructions": "Thank the customer and wish them well.",
				"max_tokens": 100,
				"temperature": 0.7
			},
			"tools": [],
			"transitions": [],
			"preserve_chat_context": false
		}
	]
}
```

### Best Practices

1. **Agent Specialization**

   - Each node should have a clear, focused role
   - Use system_prompt to define personality
   - Use instructions for specific tasks

2. **Context Management**

   - Set `preserve_chat_context: false` for routing nodes
   - Set `preserve_chat_context: true` for conversational nodes
   - Fresh context can improve routing accuracy

3. **Transition Design**

   - Prefer intent-based over regex for flexibility
   - Always provide a path back to welcome for re-routing
   - Include hangup transitions from all nodes

4. **Prompt Structure**
   - `system_prompt`: Who the agent is
   - `instructions`: What the agent should do
   - Keep both concise and actionable

---

## LangGraph Agent Schema

**Schema File:** `schemas/langgraph_schema.json` **Entry Point:**
`src/langgraph_agent.py` **Example Configs:**
`configs/langgraph/customer_support.json`,
`configs/langgraph/restaurant_ordering.json`

### Core Concepts

LangGraph provides state machine orchestration:

- **State** is explicitly managed across nodes
- **Conditional edges** enable dynamic routing
- **Checkpointing** allows resuming conversations
- **Most powerful** but also most complex

### Required Fields

```json
{
  "id": "graph-id",
  "name": "Graph Name",
  "version": "1.0",
  "entry_node": "welcome",           // Starting node
  "end_node": "__end__",             // Terminal node (usually "__end__")
  "nodes": [...],                    // Node definitions
  "edges": [...],                    // Direct connections
  "conditional_edges": [...]         // Conditional routing
}
```

### Node Structure

```json
{
	"id": "node_id",
	"name": "Display Name",
	"description": "Node purpose",
	"type": "router|conversational|tool|hangup|state_update",
	"system_prompt": "LLM behavior",
	"prompt_template": "Response template",
	"max_tokens": 150,
	"temperature": 0.7,
	"tools": [],
	"preserve_context": true
}
```

#### Node Types

1. **`router`** - Decision nodes

   - Analyze state and user input
   - Set routing variables
   - Used with conditional_edges

2. **`conversational`** - Dialogue nodes

   - Extended interactions
   - Can modify state
   - Preserve conversation history

3. **`tool`** - Action nodes

   - Execute operations
   - Update state with results
   - Typically auto-advance

4. **`hangup`** - Terminal nodes

   - End the graph
   - Final messages
   - Cleanup operations

5. **`state_update`** - State management
   - Pure state modifications
   - No LLM interaction
   - Quick transitions

### Edges vs Conditional Edges

**Direct Edges** - Unconditional connections:

```json
{
	"edges": [
		["node_a", "node_b"], // Always go from A to B
		["node_b", "node_c"] // Then B to C
	]
}
```

**Conditional Edges** - Dynamic routing:

```json
{
	"conditional_edges": [
		{
			"source": "router_node",
			"condition": "route_by_intent",
			"targets": {
				"billing": "billing_agent",
				"technical": "technical_agent",
				"general": "general_agent"
			},
			"default": "general_agent"
		}
	]
}
```

#### Routing Conditions

1. **`route_by_intent`** - Intent-based routing

   - Analyzes user input
   - Returns intent category
   - Maps to target nodes

2. **`route_by_satisfaction`** - Satisfaction check

   - Determines if user is satisfied
   - Returns "satisfied" or "more_help"
   - Common for ending conversations

3. **`route_by_state`** - State-based routing

   - Checks state variables
   - Returns state-dependent routes
   - For complex workflows

4. **`custom`** - Custom routing function
   - Define your own logic
   - Return target key
   - Maximum flexibility

### Complete Example

```json
{
	"id": "support-graph-v1",
	"name": "Support Graph",
	"version": "1.0",
	"description": "LangGraph customer support system",
	"entry_node": "welcome",
	"end_node": "__end__",
	"nodes": [
		{
			"id": "welcome",
			"name": "Welcome Router",
			"description": "Initial routing",
			"type": "router",
			"system_prompt": "You are a friendly customer service representative.",
			"prompt_template": "Welcome! I can help with billing, technical issues, or general questions. What do you need?",
			"max_tokens": 150,
			"temperature": 0.7,
			"tools": [],
			"preserve_context": false
		},
		{
			"id": "billing_agent",
			"name": "Billing Agent",
			"description": "Handle billing",
			"type": "conversational",
			"system_prompt": "You are a billing specialist.",
			"prompt_template": "I'm your billing specialist. How can I help with your account?",
			"max_tokens": 200,
			"temperature": 0.6,
			"tools": ["check_balance", "process_refund"],
			"preserve_context": true
		},
		{
			"id": "technical_agent",
			"name": "Tech Support",
			"description": "Handle technical issues",
			"type": "conversational",
			"system_prompt": "You are a technical support specialist.",
			"prompt_template": "I'm here for technical support. What issue are you experiencing?",
			"max_tokens": 200,
			"temperature": 0.6,
			"tools": ["run_diagnostics", "reset_account"],
			"preserve_context": true
		},
		{
			"id": "satisfaction_check",
			"name": "Satisfaction Router",
			"description": "Check satisfaction",
			"type": "router",
			"system_prompt": "Determine if customer needs more help.",
			"prompt_template": "Is there anything else I can help you with?",
			"max_tokens": 100,
			"temperature": 0.7,
			"tools": [],
			"preserve_context": true
		},
		{
			"id": "hangup",
			"name": "End Call",
			"description": "Goodbye",
			"type": "hangup",
			"system_prompt": "End professionally.",
			"prompt_template": "Thank you for contacting us. Have a great day!",
			"max_tokens": 100,
			"temperature": 0.7,
			"tools": [],
			"preserve_context": false
		}
	],
	"edges": [
		["billing_agent", "satisfaction_check"],
		["technical_agent", "satisfaction_check"]
	],
	"conditional_edges": [
		{
			"source": "welcome",
			"condition": "route_by_intent",
			"targets": {
				"billing": "billing_agent",
				"technical": "technical_agent",
				"hangup": "hangup"
			},
			"default": "technical_agent"
		},
		{
			"source": "satisfaction_check",
			"condition": "route_by_satisfaction",
			"targets": {
				"satisfied": "hangup",
				"more_help": "welcome"
			},
			"default": "hangup"
		}
	]
}
```

### Best Practices

1. **State Design**

   - Keep state minimal and focused
   - Use clear variable names
   - Initialize all state variables

2. **Edge Configuration**

   - Use edges for guaranteed paths
   - Use conditional_edges for decisions
   - Always provide default routes

3. **Router Nodes**

   - Keep prompts minimal in routers
   - Focus on collecting routing information
   - Don't preserve context in routers

4. **Conversational Nodes**

   - Preserve context for continuity
   - Use tools for external operations
   - Can modify state as needed

5. **Graph Flow**
   - Design clear entry and exit paths
   - Avoid loops without exit conditions
   - Test all conditional branches

---

## Comparison & Choosing a Schema

### Feature Comparison

| Feature              | Multi-Node             | Workflow           | LangGraph              |
| -------------------- | ---------------------- | ------------------ | ---------------------- |
| **Complexity**       | High                   | Low                | Medium-High            |
| **State Management** | Context variables      | Chat context       | Full state machine     |
| **Routing**          | Intent/Regex/Variable  | Intent/Regex       | Conditional edges      |
| **Agent Isolation**  | Nodes share state      | Independent agents | Nodes + state          |
| **Checkpointing**    | No                     | No                 | Yes                    |
| **Learning Curve**   | Medium                 | Easy               | Steep                  |
| **Best For**         | Complex decision trees | Simple workflows   | Advanced orchestration |

### Decision Guide

**Choose Multi-Node when:**

- You need complex conditional logic
- Variables drive transitions
- State machine patterns fit naturally
- You have existing state machine designs

**Choose Workflow when:**

- You want simple agent handoffs
- Each agent has a clear specialty
- Context preservation is straightforward
- You're new to voice agents

**Choose LangGraph when:**

- You need explicit state management
- Conversation resumption is required
- Complex routing logic is necessary
- You need maximum control and flexibility

### Migration Path

1. **Start with Workflow** for prototyping
2. **Move to Multi-Node** for more complexity
3. **Adopt LangGraph** for production-grade systems

---

## Best Practices

### General Guidelines

1. **Start Simple**

   - Begin with minimal nodes
   - Add complexity incrementally
   - Test each addition

2. **Clear Node Boundaries**

   - Each node should have one clear purpose
   - Avoid mixing concerns
   - Keep prompts focused

3. **Transition Design**

   - Provide multiple paths from each node
   - Always include error/fallback routes
   - Test edge cases

4. **Prompt Engineering**

   - Be specific and concise
   - Include relevant context
   - Test with real user inputs

5. **Tool Integration**
   - Keep tools focused and atomic
   - Handle errors gracefully
   - Return structured data

### Testing Strategy

1. **Unit Testing**

   - Test individual node prompts
   - Verify transition conditions
   - Check tool invocations

2. **Integration Testing**

   - Test full conversation flows
   - Verify context preservation
   - Check state transitions

3. **User Testing**
   - Test with real users
   - Monitor conversation quality
   - Iterate based on feedback

### Configuration Management

1. **Version Control**

   - Use semantic versioning
   - Document changes in descriptions
   - Keep old versions for rollback

2. **Environment-Specific Configs**

   - Development: verbose logging, test tools
   - Staging: production-like but safe
   - Production: optimized, monitored

3. **Schema Validation**
   - Validate configs against schemas
   - Use IDE schema support
   - Catch errors before deployment

### Performance Optimization

1. **Token Management**

   - Set appropriate `max_tokens`
   - Reduce temperature for deterministic responses
   - Use shorter prompts when possible

2. **Context Size**

   - Limit preserved history
   - Summarize long conversations
   - Clear context when changing topics

3. **Tool Efficiency**
   - Cache tool results when possible
   - Batch operations
   - Use async/parallel execution

---

## Schema Validation

All three schemas can be validated using standard JSON Schema validators:

### Command Line (using `ajv-cli`)

```bash
# Install validator
npm install -g ajv-cli

# Validate multi-node config
ajv validate -s schemas/multinode_schema.json -d configs/ecommerce_multinode.json

# Validate workflow config
ajv validate -s schemas/workflow_schema.json -d configs/workflows/customer_support.json

# Validate langgraph config
ajv validate -s schemas/langgraph_schema.json -d configs/langgraph/customer_support.json
```

### VS Code Integration

Add to `.vscode/settings.json`:

```json
{
	"json.schemas": [
		{
			"fileMatch": ["configs/*.json"],
			"url": "./schemas/multinode_schema.json"
		},
		{
			"fileMatch": ["configs/workflows/*.json"],
			"url": "./schemas/workflow_schema.json"
		},
		{
			"fileMatch": ["configs/langgraph/*.json"],
			"url": "./schemas/langgraph_schema.json"
		}
	]
}
```

This enables:

- Autocomplete for config fields
- Inline validation errors
- Hover documentation
- Field suggestions

---

## Additional Resources

- **LiveKit Agents Docs**: https://docs.livekit.io/agents/
- **LangGraph Docs**: https://langchain-ai.github.io/langgraph/
- **JSON Schema Docs**: https://json-schema.org/
- **Example Configs**: See `configs/` directory
- **Implementation Code**: See `src/agents/` and `src/workflows/` and
  `src/langgraph_flows/`

---

## Support & Contributing

For questions, issues, or contributions:

1. Check existing documentation
2. Review example configurations
3. Test with provided schemas
4. Open issues for bugs or feature requests

---

**Last Updated:** November 2025 **Schema Version:** 1.0 **Compatible With:**
LiveKit Agents v1.2.18+
