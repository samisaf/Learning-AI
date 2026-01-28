"""
This script implements a research assistant agent using the LangChain and LangGraph libraries.

The agent is designed to answer questions and perform lookups using a search engine tool (TavilySearch).

The agent's logic is structured as a
state machine using LangGraph, allowing it to reason about when to call tools and when to
respond to the user.

Key Features:
-   **Conversational Memory:** The agent maintains conversation history using a SQLite
    database as a checkpointer, allowing it to remember past interactions within a session.
-   **Tool Use:** The agent is equipped with a TavilySearch tool to find information online.
-   **Flexible LLM Backend:** The agent can be configured to use different Large Language
    Models. It is currently set up to use a local model via `ChatOllama`, but can be
    easily switched to use OpenAI's models (`ChatOpenAI`).
"""

import logging
import operator
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph

_ = load_dotenv()

logging.basicConfig(level=logging.INFO)

DEFAULT_PROMPT = """You are a smart research assistant. Use the search engine to look up information. \
You are allowed to make multiple calls (either together or in sequence). \
Only look up information when you are sure of what you want. \
If you need to look up some information before asking a follow up question, you are allowed to do that!
"""


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]


class Agent:
    def __init__(self, model, tools, system="", checkpointer=None):
        logging.info("Initializing agent")
        self.system = system
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_llm)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges(
            "llm", self.exists_action, {True: "action", False: END}
        )
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        self.graph = graph.compile(checkpointer=checkpointer)
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

    def __call__(self, messages, thread_id="1"):
        config = {"configurable": {"thread_id": thread_id}}
        return self.graph.invoke({"messages": messages}, config)

    def exists_action(self, state: AgentState):
        result = state["messages"][-1]
        return len(result.tool_calls) > 0

    def call_llm(self, state: AgentState):
        logging.info("Calling LLM")
        messages = state["messages"]
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        message = self.model.invoke(messages)
        return {"messages": [message]}

    def take_action(self, state: AgentState):
        logging.info("Taking action")
        tool_calls = state["messages"][-1].tool_calls
        results = []
        for t in tool_calls:
            print(f"Calling: {t}")
            if t["name"] not in self.tools:  # check for bad tool name from LLM
                print("\n ....bad tool name....")
                result = "bad tool name, retry"  # instruct LLM to retry if bad
            else:
                result = self.tools[t["name"]].invoke(t["args"])
            results.append(
                ToolMessage(tool_call_id=t["id"], name=t["name"], content=str(result))
            )
        print("Back to the model!")
        return {"messages": results}


if __name__ == "__main__":
    # Toggle comment to change model
    # model = ChatOllama(model="llama3.2")
    model = ChatOpenAI(model="gpt-4o-mini")

    tool = TavilySearch(max_results=2)

    # This will create 'checkpoints.sqlite' in your current directory
    with SqliteSaver.from_conn_string("checkpoints.sqlite") as memory:
        abot = Agent(model, [tool], system=DEFAULT_PROMPT, checkpointer=memory)
        # Invoke the agent
        while True:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            messages = [HumanMessage(content=user_input)]
            result = abot(messages, thread_id="1")
            print(f"Agent: {result['messages'][-1].content}")
