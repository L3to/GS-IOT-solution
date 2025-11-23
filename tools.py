import inspect
import os
from typing import Callable, Dict, List

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

from llm_config import client, llm
from rag_functions import (
    chunks_formation,
    collection_embeding,
    embed_model,
    expand_query,
    load_file_contents,
)

tool_registry: Dict[str, Callable] = {}
tool_info_registry: List[Dict[str, str]] = []


def load_prompt(prompt_name: str) -> str:
    """Carrega um prompt do diretório de prompts"""
    prompt_path = os.path.join("prompts", f"{prompt_name}.txt")
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


def tool(func: Callable) -> Callable:
    signature = inspect.signature(func)
    docstring = func.__doc__ or ""
    params = [
        {"name": param.name, "type": param.annotation}
        for param in signature.parameters.values()
    ]
    tool_info = {"name": func.__name__, "description": docstring, "parameters": params}
    tool_registry[func.__name__] = func
    tool_info_registry.append(tool_info)
    return func


@tool
def leave_chat():
    """
    Encerra a sessão de chat.

    Esta função é invocada quando o usuário deseja sair da conversa.

    Returns:
        str: Mensagem de confirmação indicando que o chat foi encerrado.
    """
    print("Exiting chat session.")
    exit(0)


@tool
def analyze_contract(
    file_path: str, question: str = "Faça uma análise completa do contrato", history: str = ""
) -> str:
    """
    Realiza análise completa de contratos e documentos jurídicos.

    Extrai metadados, identifica riscos, detecta cláusulas ausentes, mapeia obrigações e gera resumo executivo.

    Args:
        file_path (str): Caminho do arquivo do contrato a ser analisado
        question (str): Pergunta específica ou "análise completa" (padrão)
        history (str): Histórico da conversa para contexto adicional
    Returns:
        str: Análise estruturada em JSON contendo metadados, riscos, cláusulas ausentes, obrigações e resumo executivo
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            document_content = file.read()
    except Exception as e:
        return f'{{"error": "Erro ao ler o arquivo {file_path}: {e}"}}'

    template = load_prompt("analyze_contract")
    prompt = PromptTemplate.from_template(template)
    llm_chain = prompt | llm | StrOutputParser()
    generation = llm_chain.invoke({"document": document_content, "question": question, "history": history})
    return generation


@tool
def refactor_contract(
    file_path: str, question: str = "Reformule o contrato mantendo conformidade legal", history: str = ""
) -> str:
    """
    Realiza a reformulação de contratos e documentos jurídicos.

    Extrai o conteúdo do contrato fornecido e realiza a reformulação conforme a solicitação do usuário.

    Args:
        file_path (str): Caminho do arquivo do contrato a ser reformulado
        question (str): Instruções específicas para a reformulação do contrato
        history (str): Histórico da conversa para contexto adicional
    Returns:
        str: Contrato reformulado conforme as instruções fornecidas
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            document_content = file.read()
    except Exception as e:
        return f'{{"error": "Erro ao ler o arquivo {file_path}: {e}"}}'

    template = load_prompt("refactor_contract")
    prompt = PromptTemplate.from_template(template)
    llm_chain = prompt | llm | StrOutputParser()
    generation = llm_chain.invoke({"document": document_content, "question": question, "history": history})
    return generation


@tool
def generate_contracts(
    question: str, history: str
) -> str:
    """
    """
    contracts_collection = client.get_or_create_collection("contracts_collection")
    if contracts_collection.count() == 0:
        contract_documents = []
        contract_metadata = []
        contract_ids = []

        for filename in os.listdir("rag_files/contracts"):
            file_path = f"rag_files/contracts/{filename}"

            if os.path.isdir(file_path):
                continue

            contract_content = load_file_contents(file_path)
            contract_documents.append(contract_content)
            
            contract_metadata.append(
                {"source": file_path}   
            )
            contract_ids.append(f"contract_{len(contract_documents)}")

        embeddings = [embed_model.embed_query(doc) for doc in contract_documents]
        contracts_collection.add(
            documents=contract_documents,
            embeddings=embeddings,
            metadatas=contract_metadata,
            ids=contract_ids,
        )


@tool
def retrieve_brazilian_law_context_and_answer(question: str, history: str) -> str:
    """
    Realiza busca e geração de resposta (RAG) sobre legislação brasileira.

    Busca trechos relevantes da legislação brasileira usando expansão de query e similaridade vetorial,
    depois gera uma resposta fundamentada com o assistente jurídico ContratAI.

    Args:
        question (str): Pergunta jurídica a ser respondida

    Returns:
        str: Resposta jurídica gerada baseada no contexto recuperado das leis brasileiras
    """
    laws_collection = client.get_or_create_collection("laws_collection")
    if laws_collection.count() == 0:
        law_rag_raw_path = []
        law_rag_raw_content = []

        for i in os.listdir("rag_files/laws"):
            law_rag_raw_path.append(f"rag_files/laws/{i}")
            law_rag_raw_content.append(load_file_contents(law_rag_raw_path[-1]))

        law_text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\nArt.", "\n\n", ". "],
        )
        metadata_chunks = []
        chunks_formation(
            law_rag_raw_path, law_rag_raw_content, law_text_splitter, metadata_chunks
        )
        collection_embeding(laws_collection, metadata_chunks, "law")
    expanded_results = expand_query(question, laws_collection)
    query_embedding = embed_model.embed_query(question)
    results = laws_collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
    )
    relevant_docs = results["documents"][0]
    for i in expanded_results:
        for j in i["documents"][0]:
            if j not in relevant_docs:
                relevant_docs.append(j)

    combined_context = "\n\n".join(relevant_docs)
    print(combined_context)
    template = load_prompt("retrieve_law")
    prompt = PromptTemplate.from_template(template)
    llm_chain = prompt | llm | StrOutputParser()
    generation = llm_chain.invoke({"context": combined_context, "question": question, "history": history})
    print(generation)
    return generation
