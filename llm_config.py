"""
Singleton LLM Configuration Module

This module provides a centralized LLM instance to avoid circular imports
and ensure consistent model configuration across the application.
"""

import chromadb
from langchain_ollama import ChatOllama

llm = ChatOllama(model="mistral-nemo:12b", format="json", temperature=0)
client = chromadb.PersistentClient(path="chroma_db_laws")

