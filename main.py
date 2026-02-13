from dotenv import load_dotenv
import os

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
# Import Ollama fetches here - might last longer than Gemini
from pydantic_core.core_schema import model_field
from langchain.tools import tool
from langchain.agents import create_openai_tools_agent,  AgentExecutor
from todoist_api_python.api import TodoistAPI

load_dotenv()

todoist_api_key = os.getenv("TODOIST_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")
ollama_api_key = os.getenv("OLLAMA_API_KEY")

todoist = TodoistAPI(todoist_api_key)

@tool
def add_task(task, desc=None):
    # todoist api code here
    """Add a new task. Use this when the user wants to add or create a task""" # Value error raised without this text
    todoist.add_task(content=task,
                     description=desc)

@tool
def show_tasks():
    """Show all tasks from Todoist. Use this tool when the user wants to see their tasks."""
    results_paginator = todoist.get_tasks()
    tasks = []
    for task_list in results_paginator:
        for task in task_list:
            tasks.append(task.content)
    return tasks


tools = [add_task, show_tasks]

llm = ChatGoogleGenerativeAI(
    model='gemini-2.5-flash',
    google_api_key=gemini_api_key,
    temperature=0.3  # closer to 0 is more deterministic, higher values are more creative
)

#system_prompt = "You are a philosopher and life coach. You will help me understand myself better."
system_prompt = """You are a helpful assistant.
You will help the user add tasks.
You will help the user show existing tasks.
"""

prompt = ChatPromptTemplate([
    ("system", system_prompt),
    ("user", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
    MessagesPlaceholder("history")
])

# chain = prompt | llm | StrOutputParser()
agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

#response = chain.invoke({"input":user_input})

history = []
while True:
    user_input = input("You: ")
    response = agent_executor.invoke({"input": user_input, "history": history})
    print(response['output'])
    history.append(HumanMessage(content=user_input))
    history.append(AIMessage(content=response['output']))
