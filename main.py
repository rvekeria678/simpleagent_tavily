from dotenv import load_dotenv
from typing import Annotated, Literal
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import tools_condition, ToolNode
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_tavily import TavilySearch

load_dotenv()
@tool
def search(query: str) -> list:
    """
    Searches a query through a search engine
    Args:
        query: prompt used to search results
    """
    print("(Query Search Tool Invoked)")
    results = TavilySearch(max_results=2).invoke(query)
    return [result["content"] for result in results]

tools = [search]

llm = init_chat_model("gpt-3.5-turbo")
llm_with_tools = llm.bind_tools(tools)

sys_msg = SystemMessage(content="You are a helpful research assistant that finds accurate, relevant, and credible information in response to user questions..")

# Node
def assistant(state: MessagesState):
    messages = [sys_msg] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": state["messages"] + [response]}

# Build Graph
builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))
builder.set_entry_point("assistant")

builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")
graph = builder.compile()

# Run
user_input = input(">> ")
initial_state = {"messages": HumanMessage(content=user_input)}
final_state = graph.invoke(initial_state)

for msg in final_state["messages"]:
    if msg.type == "ai":
        print(f"Assistant: {msg.content}")