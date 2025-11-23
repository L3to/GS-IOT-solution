"""
Singleton LLM Configuration Module

This module provides a centralized LLM instance to avoid circular imports
and ensure consistent model configuration across the application.
"""

import os
import chromadb
from langchain_ollama import ChatOllama

llm = ChatOllama(model="qwen3-coder:latest", format="json", temperature=0)
client = chromadb.PersistentClient(path="chroma_db")

def load_prompt(prompt_name: str) -> str:
    """Carrega um prompt do diretório de prompts"""
    prompt_path = os.path.join("prompts", f"{prompt_name}.txt")
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()