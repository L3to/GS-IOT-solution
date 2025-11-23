import json
import os
from datetime import datetime
from typing import List, Optional
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pymongo import MongoClient

from main import app as agent_app
from main import tools_list
from tools import tool_info_registry

load_dotenv()

uri = f"mongodb+srv://fiap_admin:{os.environ.get('DB_PASSWORD')}@fiap.fzp5f22.mongodb.net/?appName=FIAP"
db_name = "contratai_db"
mongodb_client = MongoClient(uri)
db = mongodb_client[db_name]
sessions_collection = db["chat_sessions"]

api = FastAPI(
    title="ContratAI API",
    description="API para gerenciar sessões de chat e executar agente jurídico",
    version="1.0.0",
)

api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Message(BaseModel):
    timestamp: datetime
    role: str
    content: str


class Session(BaseModel):
    session_id: str
    created_at: datetime
    updated_at: datetime
    messages: List[Message]


class QuestionRequest(BaseModel):
    question: str
    session_id: Optional[str] = None


class QuestionResponse(BaseModel):
    session_id: str
    answer: str
    tool_used: Optional[str] = None


class UpdateSessionRequest(BaseModel):
    messages: List[Message]


def clip_history(history: str, max_chars: int = 8000) -> str:
    if len(history) > max_chars:
        return history[-max_chars:]
    return history


def save_to_mongodb(session_id: str, role: str, content: str):
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


@api.get("/")
def root():
    return {
        "message": "ContratAI API",
        "version": "1.0.0",
        "endpoints": {
            "GET /sessions": "Listar todas as sessões",
            "GET /sessions/{session_id}": "Obter sessão específica",
            "POST /sessions": "Criar nova sessão",
            "PUT /sessions/{session_id}": "Atualizar sessão",
            "DELETE /sessions/{session_id}": "Deletar sessão",
            "POST /chat": "Enviar pergunta para o agente",
            "GET /tools": "Listar ferramentas disponíveis",
        },
    }


@api.get("/sessions", response_model=List[dict])
def get_all_sessions():
    """Lista todas as sessões de chat"""
    sessions = list(sessions_collection.find({}, {"_id": 0}))
    return sessions


@api.get("/sessions/{session_id}")
def get_session(session_id: str):
    """Obtém uma sessão específica por ID"""
    session = sessions_collection.find_one({"session_id": session_id}, {"_id": 0})
    if not session:
        raise HTTPException(status_code=404, detail="Sessão não encontrada")
    return session


@api.post("/sessions")
def create_session():
    """Cria uma nova sessão de chat"""
    session_id = str(uuid4())[:8]
    session_data = {
        "session_id": session_id,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
        "messages": [],
    }
    sessions_collection.insert_one(session_data)
    return {"session_id": session_id, "message": "Sessão criada com sucesso"}


@api.put("/sessions/{session_id}")
def update_session(session_id: str, data: UpdateSessionRequest):
    """Atualiza as mensagens de uma sessão"""
    session = sessions_collection.find_one({"session_id": session_id})
    if not session:
        raise HTTPException(status_code=404, detail="Sessão não encontrada")

    messages_dict = [msg.dict() for msg in data.messages]

    result = sessions_collection.update_one(
        {"session_id": session_id},
        {"$set": {"messages": messages_dict, "updated_at": datetime.utcnow()}},
    )

    if result.modified_count == 0:
        raise HTTPException(status_code=400, detail="Falha ao atualizar sessão")

    return {"message": "Sessão atualizada com sucesso"}


@api.delete("/sessions/{session_id}")
def delete_session(session_id: str):
    """Delete uma sessão de chat"""
    result = sessions_collection.delete_one({"session_id": session_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Sessão não encontrada")
    return {"message": "Sessão deletada com sucesso"}


@api.post("/chat", response_model=QuestionResponse)
def chat(request: QuestionRequest):
    """Envia uma pergunta para o agente ContratAI"""
    session_id = request.session_id or str(uuid4())[:8]

    save_to_mongodb(session_id, "user", request.question)

    full_history = load_from_mongodb(session_id)

    initial_state = {
        "question": request.question,
        "history": full_history,
        "use_tool": False,
        "tool_exec": "",
        "tools_list": tools_list,
        "session_id": session_id,
    }

    final_state = agent_app.invoke(initial_state)

    answer = ""
    tool_used = None

    if final_state.get("tool_exec"):
        try:
            tool_data = json.loads(final_state["tool_exec"])
            tool_used = tool_data.get("function")
        except Exception:
            pass

    session = sessions_collection.find_one({"session_id": session_id})
    if session and "messages" in session and len(session["messages"]) > 0:
        last_message = session["messages"][-1]
        if last_message["role"] == "assistant":
            answer = last_message["content"]

    return QuestionResponse(session_id=session_id, answer=answer, tool_used=tool_used)


@api.get("/tools")
def get_tools():
    """Lista todas as ferramentas disponíveis"""
    return {
        "tools": [
            {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool.get("parameters", []),
            }
            for tool in tool_info_registry
        ]
    }


@api.delete("/sessions")
def delete_all_sessions():
    """Deleta todas as sessões (cuidado!)"""
    result = sessions_collection.delete_many({})
    return {"message": f"{result.deleted_count} sessões deletadas"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(api, host="0.0.0.0", port=8000)
