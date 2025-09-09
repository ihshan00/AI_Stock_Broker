import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, TypedDict, Annotated
from dataclasses import dataclass
from enum import Enum

from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


# =============================================================================
# Data Models and Types
# =============================================================================

class AgentType(Enum):
    PLAN_AGENT = "plan_agent"
    BILLING_AGENT = "billing_agent"
    USAGE_AGENT = "usage_agent"
    SERVICE_AGENT = "service_agent"
    GENERAL_AGENT = "general_agent"


@dataclass
class UserData:
    user_id: str
    billing: Dict
    plan: Dict
    usage: Dict


class CustomerAssistantState(TypedDict):
    messages: Annotated[List, "The conversation messages"]
    user_id: str
    user_data: Optional[UserData]
    selected_agent: Optional[AgentType]
    query_intent: Optional[str]
    tools_called: List[str]
    response: Optional[str]
    requires_escalation: bool


# =============================================================================
# Database Layer (Simulated)
# =============================================================================

class CustomerDataStore:
    """Simulated aggregated customer data store"""
    
    def __init__(self):
        # Simulate the aggregated view you described
        self.user_data_cache = {
            "user_123": {
                "billing": {
                    "latest": {
                        "id": "inv-222",
                        "amount_cents": 14900,
                        "status": "open"
                    },
                    "payment_method": "card_ending_1234",
                    "next_billing_date": "2025-10-15"
                },
                "plan": {
                    "name": "Plus",
                    "expiry_date": "2025-09-23",
                    "features": ["api_access", "premium_support", "analytics"]
                },
                "usage": {
                    "last_7d": {"units_total": 301},
                    "last_30d": {"units_total": 1234},
                    "current_month": {"units_total": 856}
                }
            }
        }
    
    async def get_user_data(self, user_id: str) -> Optional[UserData]:
        """Fetch aggregated user data"""
        data = self.user_data_cache.get(user_id)
        if data:
            return UserData(
                user_id=user_id,
                billing=data["billing"],
                plan=data["plan"],
                usage=data["usage"]
            )
        return None


# =============================================================================
# Specialized Agent Tools
# =============================================================================

@tool
def get_plan_expiry_info(user_data: dict) -> dict:
    """Get detailed plan expiry information and recommendations"""
    plan = user_data.get("plan", {})
    expiry_date_str = plan.get("expiry_date")
    
    if not expiry_date_str:
        return {"error": "No plan expiry date found"}
    
    expiry_date = datetime.strptime(expiry_date_str, "%Y-%m-%d")
    today = datetime.now()
    days_until_expiry = (expiry_date - today).days
    
    # Generate recommendations based on current plan
    recommendations = []
    if days_until_expiry <= 7:
        recommendations = [
            {"plan": "Plus Renewal", "discount": "10%", "features": "Same features"},
            {"plan": "Pro Upgrade", "discount": "15%", "features": "Additional API limits"}
        ]
    
    return {
        "current_plan": plan.get("name"),
        "expiry_date": expiry_date_str,
        "days_until_expiry": days_until_expiry,
        "status": "expiring_soon" if days_until_expiry <= 30 else "active",
        "recommendations": recommendations
    }


@tool
def get_billing_details(user_data: dict) -> dict:
    """Get comprehensive billing information"""
    billing = user_data.get("billing", {})
    latest_invoice = billing.get("latest", {})
    
    return {
        "latest_invoice": {
            "id": latest_invoice.get("id"),
            "amount": f"${latest_invoice.get('amount_cents', 0) / 100:.2f}",
            "status": latest_invoice.get("status"),
        },
        "payment_method": billing.get("payment_method"),
        "next_billing_date": billing.get("next_billing_date"),
        "billing_history_available": True
    }


@tool
def get_usage_analytics(user_data: dict) -> dict:
    """Get detailed usage analytics and insights"""
    usage = user_data.get("usage", {})
    plan = user_data.get("plan", {})
    
    # Calculate usage trends
    last_7d = usage.get("last_7d", {}).get("units_total", 0)
    last_30d = usage.get("last_30d", {}).get("units_total", 0)
    
    daily_average = last_7d / 7
    monthly_projection = daily_average * 30
    
    return {
        "last_7_days": last_7d,
        "last_30_days": last_30d,
        "daily_average": round(daily_average, 2),
        "monthly_projection": round(monthly_projection, 0),
        "current_plan": plan.get("name"),
        "usage_trend": "increasing" if last_7d > (last_30d / 4) else "stable"
    }


@tool
def check_service_eligibility(user_data: dict, service_type: str) -> dict:
    """Check if user is eligible for service changes"""
    plan = user_data.get("plan", {})
    billing = user_data.get("billing", {})
    
    return {
        "service_type": service_type,
        "eligible": billing.get("latest", {}).get("status") == "paid",
        "current_plan": plan.get("name"),
        "restrictions": [] if billing.get("latest", {}).get("status") == "paid" else ["Outstanding balance"]
    }


# =============================================================================
# Agent Implementations
# =============================================================================

class PlanningAgent:
    """LLM-based agent router - decides which specialized agent to use"""
    
    def __init__(self, llm):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an intelligent routing agent for a customer service system. 
            Your job is to analyze customer queries and determine which specialized agent should handle the request.

            Available agents:
            - PLAN_AGENT: Handles plan expiry, renewals, upgrades, plan features
            - BILLING_AGENT: Handles invoices, payments, billing history, payment methods
            - USAGE_AGENT: Handles usage statistics, analytics, limits, consumption patterns
            - SERVICE_AGENT: Handles account changes, service activation/deactivation
            - GENERAL_AGENT: Handles general inquiries, greetings, non-specific questions

            Analyze the user query and respond with ONLY the agent name (e.g., "PLAN_AGENT").
            Consider the primary intent of the user's question."""),
            ("human", "Customer query: {query}")
        ])
    
    async def route_query(self, query: str) -> AgentType:
        """Route query to appropriate agent using LLM"""
        messages = self.prompt.format_messages(query=query)
        response = await self.llm.ainvoke(messages)
        
        agent_mapping = {
            "PLAN_AGENT": AgentType.PLAN_AGENT,
            "BILLING_AGENT": AgentType.BILLING_AGENT,
            "USAGE_AGENT": AgentType.USAGE_AGENT,
            "SERVICE_AGENT": AgentType.SERVICE_AGENT,
            "GENERAL_AGENT": AgentType.GENERAL_AGENT
        }
        
        agent_name = response.content.strip().upper()
        return agent_mapping.get(agent_name, AgentType.GENERAL_AGENT)


class SpecializedAgent:
    """Base class for specialized agents"""
    
    def __init__(self, agent_type: AgentType, llm, tools: List):
        self.agent_type = agent_type
        self.llm = llm
        self.tools = tools
        self.tool_node = ToolNode(tools)
    
    async def process_query(self, state: CustomerAssistantState) -> Dict:
        """Process query using specialized knowledge"""
        user_data_dict = {
            "billing": state["user_data"].billing,
            "plan": state["user_data"].plan,
            "usage": state["user_data"].usage
        } if state["user_data"] else {}
        
        # Create specialized system prompt based on agent type
        system_prompts = {
            AgentType.PLAN_AGENT: f"""You are a plan specialist. Help users with plan-related queries using available tools.
            User Data: {json.dumps(user_data_dict, indent=2)}
            Use the get_plan_expiry_info tool to provide detailed plan information.""",
            
            AgentType.BILLING_AGENT: f"""You are a billing specialist. Help users with billing and payment queries.
            User Data: {json.dumps(user_data_dict, indent=2)}
            Use the get_billing_details tool to provide comprehensive billing information.""",
            
            AgentType.USAGE_AGENT: f"""You are a usage analytics specialist. Help users understand their usage patterns.
            User Data: {json.dumps(user_data_dict, indent=2)}
            Use the get_usage_analytics tool to provide detailed usage insights.""",
            
            AgentType.SERVICE_AGENT: f"""You are a service specialist. Help users with account and service changes.
            User Data: {json.dumps(user_data_dict, indent=2)}
            Use the check_service_eligibility tool to verify service options.""",
            
            AgentType.GENERAL_AGENT: """You are a general customer service agent. Provide helpful responses for general inquiries."""
        }
        
        messages = [
            SystemMessage(content=system_prompts[self.agent_type]),
            HumanMessage(content=state["messages"][-1].content)
        ]
        
        # Let the LLM decide if it needs to use tools
        response = await self.llm.bind_tools(self.tools).ainvoke(messages)
        
        # If tools are called, execute them
        if response.tool_calls:
            tool_results = []
            for tool_call in response.tool_calls:
                if tool_call["name"] in ["get_plan_expiry_info", "get_billing_details", 
                                       "get_usage_analytics", "check_service_eligibility"]:
                    # Pass user_data to tools that need it
                    tool_call["args"]["user_data"] = user_data_dict
                
                # Execute tool
                tool_result = await self._execute_tool(tool_call, user_data_dict)
                tool_results.append(tool_result)
            
            # Generate final response with tool results
            final_messages = messages + [response] + tool_results
            final_response = await self.llm.ainvoke(final_messages)
            return {"response": final_response.content, "tools_used": [tc["name"] for tc in response.tool_calls]}
        
        return {"response": response.content, "tools_used": []}
    
    async def _execute_tool(self, tool_call: Dict, user_data: Dict):
        """Execute tool call and return result"""
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        
        if tool_name == "get_plan_expiry_info":
            result = get_plan_expiry_info.invoke(tool_args)
        elif tool_name == "get_billing_details":
            result = get_billing_details.invoke(tool_args)
        elif tool_name == "get_usage_analytics":
            result = get_usage_analytics.invoke(tool_args)
        elif tool_name == "check_service_eligibility":
            result = check_service_eligibility.invoke(tool_args)
        else:
            result = {"error": f"Unknown tool: {tool_name}"}
        
        return AIMessage(content=f"Tool result: {json.dumps(result, indent=2)}")


# =============================================================================
# LangGraph Workflow
# =============================================================================

class CustomerAssistantWorkflow:
    def __init__(self):
        self.data_store = CustomerDataStore()
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.planning_agent = PlanningAgent(self.llm)
        
        # Initialize specialized agents
        self.agents = {
            AgentType.PLAN_AGENT: SpecializedAgent(
                AgentType.PLAN_AGENT, self.llm, [get_plan_expiry_info]
            ),
            AgentType.BILLING_AGENT: SpecializedAgent(
                AgentType.BILLING_AGENT, self.llm, [get_billing_details]
            ),
            AgentType.USAGE_AGENT: SpecializedAgent(
                AgentType.USAGE_AGENT, self.llm, [get_usage_analytics]
            ),
            AgentType.SERVICE_AGENT: SpecializedAgent(
                AgentType.SERVICE_AGENT, self.llm, [check_service_eligibility]
            ),
            AgentType.GENERAL_AGENT: SpecializedAgent(
                AgentType.GENERAL_AGENT, self.llm, []
            )
        }
        
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(CustomerAssistantState)
        
        # Add nodes
        workflow.add_node("identify_user", self.identify_user)
        workflow.add_node("route_query", self.route_query)
        workflow.add_node("execute_agent", self.execute_agent)
        workflow.add_node("validate_response", self.validate_response)
        
        # Add edges
        workflow.add_edge(START, "identify_user")
        workflow.add_edge("identify_user", "route_query")
        workflow.add_edge("route_query", "execute_agent")
        workflow.add_edge("execute_agent", "validate_response")
        workflow.add_edge("validate_response", END)
        
        return workflow.compile()
    
    async def identify_user(self, state: CustomerAssistantState) -> Dict:
        """Identify user and load their data"""
        # In real implementation, extract user_id from authentication context
        user_id = state.get("user_id", "user_123")  # Simulated
        user_data = await self.data_store.get_user_data(user_id)
        
        return {
            "user_data": user_data,
            "user_id": user_id
        }
    
    async def route_query(self, state: CustomerAssistantState) -> Dict:
        """Use planning agent to route the query"""
        if not state["messages"]:
            return {"selected_agent": AgentType.GENERAL_AGENT}
        
        last_message = state["messages"][-1].content
        selected_agent = await self.planning_agent.route_query(last_message)
        
        return {
            "selected_agent": selected_agent,
            "query_intent": last_message
        }
    
    async def execute_agent(self, state: CustomerAssistantState) -> Dict:
        """Execute the selected specialized agent"""
        selected_agent = state["selected_agent"]
        agent = self.agents[selected_agent]
        
        result = await agent.process_query(state)
        
        return {
            "response": result["response"],
            "tools_called": result.get("tools_used", [])
        }
    
    async def validate_response(self, state: CustomerAssistantState) -> Dict:
        """Validate and potentially enhance the response"""
        # Add validation logic here
        # For now, just pass through
        return {
            "requires_escalation": False
        }
    
    async def process_customer_query(self, query: str, user_id: str = "user_123") -> Dict:
        """Main entry point for processing customer queries"""
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "user_id": user_id,
            "user_data": None,
            "selected_agent": None,
            "query_intent": None,
            "tools_called": [],
            "response": None,
            "requires_escalation": False
        }
        
        result = await self.workflow.ainvoke(initial_state)
        
        return {
            "response": result["response"],
            "agent_used": result["selected_agent"],
            "tools_called": result["tools_called"],
            "requires_escalation": result["requires_escalation"]
        }


# =============================================================================
# Usage Example
# =============================================================================

async def main():
    """Example usage of the customer assistant"""
    assistant = CustomerAssistantWorkflow()
    
    # Example queries
    test_queries = [
        "When does my plan expire?",
        "What's my current bill amount?", 
        "How much have I used this month?",
        "I want to cancel my service",
        "Hello, how are you today?"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Query: {query}")
        result = await assistant.process_customer_query(query)
        print(f"🤖 Agent: {result['agent_used']}")
        print(f"🛠️  Tools: {result['tools_called']}")
        print(f"💬 Response: {result['response']}")
        print("-" * 80)


# Run the example
if __name__ == "__main__":
    # Install required packages:
    # pip install langgraph langchain-openai langchain-core
    
    # Set your OpenAI API key as environment variable:
    # export OPENAI_API_KEY="your-key-here"
    
    asyncio.run(main())
