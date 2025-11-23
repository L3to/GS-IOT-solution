import json
import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Literal, TypedDict
from uuid import uuid4

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.graph import END, StateGraph
from pymongo import MongoClient

from llm_config import llm
from tools import load_prompt, tool_info_registry, tool_registry

load_dotenv()
uri = f"mongodb+srv://fiap_admin:{os.environ.get('DB_PASSWORD')}@fiap.fzp5f22.mongodb.net/?appName=FIAP"
db_name = "contratai_db"

mongodb_client = MongoClient(uri)
db = mongodb_client[db_name]
sessions_collection = db["chat_sessions"]


class GeneralState(TypedDict):
    question: str
    history: str
    use_tool: bool
    tool_exec: str
    tools_list: str
    session_id: str


class AgentBase(ABC):
    def __init__(self, state: GeneralState):
        self.state = state

    @abstractmethod
    def get_prompt_template(self) -> str:
        pass

    def execute(self) -> GeneralState:
        full_history = self.state["history"]
        clipped_history = clip_history(self.state["history"])

        template = self.get_prompt_template()
        prompt = PromptTemplate.from_template(template)
        llm_chain = prompt | llm | StrOutputParser()
        generation = llm_chain.invoke(
            {
                "history": clipped_history,
                "question": self.state["question"],
                "use_tool": self.state["use_tool"],
                "tools_list": self.state["tools_list"],
            }
        )

        data = json.loads(generation)
        response_text = ""
        if "answer" in data:
            response_text = data["answer"]
            print(f"\n{response_text}\n")

        self.state["use_tool"] = data.get("use_tool", False)
        self.state["tool_exec"] = generation

        self.state["history"] = full_history + "\n" + generation

        if isinstance(self, ChatAgent) and response_text:
            save_to_mongodb(
                self.state["session_id"],
                "assistant",
                response_text,
            )

        return self.state


class ChatAgent(AgentBase):
    def get_prompt_template(self) -> str:
        return load_prompt("chat_agent")


class ToolAgent(AgentBase):
    def get_prompt_template(self) -> str:
        return load_prompt("tool_agent")


def clip_history(history: str, max_chars: int = 8000) -> str:
    """Recorta histórico apenas para o contexto do LLM (não afeta o BD)"""
    if len(history) > max_chars:
        return history[-max_chars:]
    return history


def save_to_mongodb(session_id: str, role: str, content: str):
    """Adiciona uma única mensagem ao array de mensagens da sessão"""
    sessions_collection.update_one(
        {"session_id": session_id},
        {
            "$set": {"updated_at": datetime.utcnow()},
            "$setOnInsert": {"created_at": datetime.utcnow()},
            "$push": {
                "messages": {
                    "timestamp": datetime.utcnow(),
                    "role": role,
                    "content": content,
                }
            },
        },
        upsert=True,
    )


def load_from_mongodb(session_id: str) -> str:
    """Reconstrói o histórico completo a partir do array de mensagens"""
    session = sessions_collection.find_one({"session_id": session_id})
    if session and "messages" in session:
        history = ""
        for msg in session["messages"]:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            prefix = "User" if role == "user" else "Assistant"
            history += f"\n{prefix}: {content}"
        return history
    return ""


def ToolExecutor(state: GeneralState) -> GeneralState:
    if not state["tool_exec"]:
        raise ValueError("No tool_exec data available to execute.")

    choice = json.loads(state["tool_exec"])
    tool_name = choice["function"]
    args = choice["args"]

    if tool_name not in tool_registry:
        raise ValueError(f"Tool {tool_name} not found in registry.")

    print(f"\nExecutando ferramenta: {tool_name}")
    result = tool_registry[tool_name](*args)

    try:
        result_data = json.loads(result)
        ai_response = result_data.get("answer", result)
    except Exception:
        ai_response = result

    state["history"] += f"\nExecuted {tool_name} with result: {result}"

    print(f"\nResultado da ferramenta: {ai_response}\n")

    save_to_mongodb(
        state["session_id"],
        "assistant",
        ai_response,
    )

    state["use_tool"] = False
    state["tool_exec"] = ""
    return state


def check_use_tool(state: GeneralState) -> Literal["use tool", "not use tool"]:
    if state.get("use_tool"):
        return "use tool"
    else:
        return "not use tool"


workflow = StateGraph(GeneralState)


def chat_agent_node(state: GeneralState) -> GeneralState:
    agent = ChatAgent(state)
    return agent.execute()


def tool_agent_node(state: GeneralState) -> GeneralState:
    agent = ToolAgent(state)
    return agent.execute()


workflow.add_node("chat_agent", chat_agent_node)

workflow.add_node("tool_agent", tool_agent_node)

workflow.add_node("tool_executor", ToolExecutor)

workflow.set_entry_point("chat_agent")

workflow.add_conditional_edges(
    "chat_agent",
    check_use_tool,
    {"use tool": "tool_agent", "not use tool": END},
)

workflow.add_edge("tool_agent", "tool_executor")
workflow.add_edge("tool_executor", END)

app = workflow.compile()

tools_list = json.dumps(
    [
        {"name": tool["name"], "description": tool["description"]}
        for tool in tool_info_registry
    ]
)


def question(question_text: str, session_id: str):
    """Processa uma pergunta salvando tudo no MongoDB"""
    save_to_mongodb(session_id, "user", question_text)

    full_history = load_from_mongodb(session_id)

    initial_state: GeneralState = {
        "question": question_text,
        "history": full_history,
        "use_tool": False,
        "tool_exec": "",
        "tools_list": tools_list,
        "session_id": session_id,
    }

    app.invoke(initial_state)


session_id_input = input(
    "\nDigite o ID da sessão (ou Enter para nova sessão): "
).strip()

if session_id_input:
    loaded_history = load_from_mongodb(session_id_input)
    if loaded_history:
        current_session_id = session_id_input
        msg_count = loaded_history.count("\nUser:") + loaded_history.count(
            "\nAssistant:"
        )
        print(f"\nSessão '{current_session_id}' carregada! ({msg_count} mensagens)\n")
    else:
        current_session_id = session_id_input
        print(f"\nNova sessão criada com ID: {current_session_id}\n")
else:
    current_session_id = str(uuid4())[:8]
    print(f"\nNova sessão criada com ID: {current_session_id}\n")

while True:
    r = ""
    while r.strip() == "":
        r = input()
    question(r, current_session_id)
