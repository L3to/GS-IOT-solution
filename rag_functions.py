import json
import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings

from llm_config import llm, load_prompt

embed_model = OllamaEmbeddings(model="mistral-nemo:12b")


def load_file_contents(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
            if not content.strip():
                raise ValueError("File is empty")
            return content
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None


def chunks_formation(raw_paths, raw_contents, text_splitter, metadata_chunks):
    for file_path, content in zip(raw_paths, raw_contents):
        chunks = text_splitter.split_text(content)
        for i, chunk in enumerate(chunks):
            metadata_chunks.append(
                {
                    "text": chunk,
                    "source": os.path.basename(file_path),
                    "chunk_index": i,
                }
            )


def collection_embeding(collection_name, metadata_chunks, id_prefix):
    total = len(metadata_chunks)
    for i, chunk in enumerate(metadata_chunks):
        chunk_embedding = embed_model.embed_documents([chunk["text"]])
        collection_name.add(
            documents=[chunk["text"]],
            metadatas=[
                {"source": chunk["source"], "chunk_index": chunk["chunk_index"]}
            ],
            ids=[f"{id_prefix}_chunk_{i}"],
            embeddings=chunk_embedding,
        )
        if (i + 1) % 100 == 0 or (i + 1) == total:
            percentage = ((i + 1) / total) * 100
            print(f"Progress: {i + 1}/{total} chunks ({percentage:.1f}%)")


def expand_query(question, collection):
    template = load_prompt("expand_query")
    expansion_prompt = PromptTemplate.from_template(template)
    expansion_llm_chain = expansion_prompt | llm | StrOutputParser()
    expansions = expansion_llm_chain.invoke({"question": question})
    data = json.loads(expansions)
    variations = data.get("variations", [])
    expanded_results = []
    for var in variations:
        query_embedding = embed_model.embed_query(var)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=5,
        )
        expanded_results.append((results))
    return expanded_results


def law_question_filter(question: str, history: str) -> str:
    """
    Identifica qual(is) arquivo(s) de lei contém(ém) a resposta para a pergunta.
    Retorna uma lista de nomes de arquivos em formato JSON.
    """
    filter_template = load_prompt("filter_law_files")
    filter_prompt = PromptTemplate.from_template(filter_template)
    llm = ChatOllama(model="gpt-oss:20b", format="json", temperature=0)
    filter_chain = filter_prompt | llm | StrOutputParser()
    result = filter_chain.invoke({"question": question, "history": history})

    try:
        data = json.loads(result)
        return data.get("files", [])
    except json.JSONDecodeError:
        return ""


def contract_filter(available_contracts, question, history):
    """
    Identifica qual tipo de contrato o usuário deseja gerar dentre os disponíveis.
    Retorna um dicionário com match (bool), confidence (str), contract_type (str) e reason (str).
    """
    filter_template = load_prompt("filter_contract_fields")
    filter_prompt = PromptTemplate.from_template(filter_template)
    llm_filter = ChatOllama(model="gpt-oss:20b", format="json", temperature=0)
    filter_chain = filter_prompt | llm_filter | StrOutputParser()
    result = filter_chain.invoke(
        {
            "available_contracts": available_contracts,
            "question": question,
            "history": history,
        }
    )

    try:
        data = json.loads(result)
        return data
    except json.JSONDecodeError:
        return {
            "match": False,
            "confidence": "low",
            "contract_type": "",
            "reason": "Erro ao processar validação",
        }
