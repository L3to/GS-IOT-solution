import inspect
import json
import os
from typing import Callable, Dict, List

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder

from llm_config import client, llm, load_prompt
from rag_functions import (
    chunks_formation,
    collection_embeding,
    contract_filter,
    embed_model,
    expand_query,
    law_question_filter,
    load_file_contents,
)

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

tool_registry: Dict[str, Callable] = {}
tool_info_registry: List[Dict[str, str]] = []


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
    file_path: str,
    question: str = "Faça uma análise completa do contrato",
    history: str = "",
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
    print(f"\nAnalisando contrato: {os.path.basename(file_path)}")
    print("Carregando documento...")

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            document_content = file.read()
        print(f"Documento carregado ({len(document_content)} caracteres)")
    except FileNotFoundError:
        error_msg = f"Erro: Arquivo não encontrado em '{file_path}'"
        print(error_msg)
        return json.dumps({"answer": error_msg, "error": "file_not_found"})
    except Exception as e:
        error_msg = f"Erro ao ler o arquivo: {str(e)}"
        print(error_msg)
        return json.dumps({"answer": error_msg, "error": str(e)})

    print("Executando análise jurídica profunda...")
    template = load_prompt("analyze_contract")
    prompt = PromptTemplate.from_template(template)
    llm_chain = prompt | llm | StrOutputParser()
    generation = llm_chain.invoke(
        {"document": document_content, "question": question, "history": history}
    )

    try:
        analysis_data = json.loads(generation)
        risk_level = analysis_data.get("risk_analysis", {}).get("overall_risk", "N/A")
        recommendation = analysis_data.get("executive_summary", {}).get(
            "recommendation", "N/A"
        )
        print("\nAnálise concluída!")
        print(f"   Nível de risco: {risk_level}")
        print(f"   Recomendação: {recommendation}\n")

        # Add user-friendly answer field
        analysis_data["answer"] = (
            f"Análise do contrato concluída. Nível de risco: {risk_level}. Recomendação: {recommendation}. Verifique os detalhes completos no JSON retornado."
        )
        return json.dumps(analysis_data, ensure_ascii=False)
    except Exception:
        print("Análise concluída!\n")
        return generation


@tool
def refactor_contract(
    file_path: str,
    question: str = "Reformule o contrato mantendo conformidade legal",
    history: str = "",
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
    print(f"\nReformulando contrato: {os.path.basename(file_path)}")
    print("Carregando documento original...")

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            document_content = file.read()
        print(f"Documento carregado ({len(document_content)} caracteres)")
    except FileNotFoundError:
        error_msg = f"Erro: Arquivo não encontrado em '{file_path}'"
        print(error_msg)
        return json.dumps({"answer": error_msg, "error": "file_not_found"})
    except Exception as e:
        error_msg = f"Erro ao ler o arquivo: {str(e)}"
        print(error_msg)
        return json.dumps({"answer": error_msg, "error": str(e)})

    print(f"Processando reformulação: {question}")
    template = load_prompt("refactor_contract")
    prompt = PromptTemplate.from_template(template)
    llm_chain = prompt | llm | StrOutputParser()
    generation = llm_chain.invoke(
        {"document": document_content, "question": question, "history": history}
    )

    try:
        data = json.loads(generation)
        refactored_data = data.get("refactored_contract", {})
        full_text = refactored_data.get("full_text", "")

        if not full_text:
            print("Aviso: Contrato reformulado está vazio")
            data["answer"] = (
                "Erro ao reformular o contrato. O texto reformulado está vazio."
            )
            return json.dumps(data, ensure_ascii=False)

        output_path = "refactored_contract.txt"
        with open(output_path, "w", encoding="utf-8") as f_out:
            f_out.write(full_text)

        print("\nContrato reformulado com sucesso!")
        print(f"   Salvo em: {output_path}")
        print(f"   Tamanho: {len(full_text)} caracteres\n")

        # Add user-friendly answer
        data["answer"] = (
            f"Contrato reformulado com sucesso! O novo contrato foi salvo em '{output_path}' com {len(full_text)} caracteres. Principais alterações: {refactored_data.get('summary', 'Veja o documento completo.')}"
        )
        return json.dumps(data, ensure_ascii=False)
    except json.JSONDecodeError:
        error_msg = "Erro ao processar resposta da reformulação"
        print(error_msg)
        return json.dumps({"answer": error_msg, "error": "json_decode_error"})
    except Exception as e:
        error_msg = f"Erro ao salvar contrato reformulado: {str(e)}"
        print(error_msg)
        return json.dumps({"answer": error_msg, "error": str(e)})


@tool
def generate_contracts(question: str, history: str) -> str:
    """
    Gera contratos personalizados usando templates da base RAG.

    Identifica o tipo de contrato, pede informações necessárias e preenche o template.

    Args:
        question (str): Solicitação de geração de contrato ou dados para preenchimento
        history (str): Histórico da conversa

    Returns:
        str: JSON com campos necessários OU contrato preenchido
    """
    print("\nGerando contrato personalizado...")

    with open("contract_fields_mapping.json", "r", encoding="utf-8") as f:
        fields_mapping = json.load(f)

    contracts_collection = client.get_or_create_collection("contracts_collection")
    if contracts_collection.count() == 0:
        print("Inicializando base de templates de contratos...")
        contract_documents = []
        contract_metadata = []
        contract_ids = []

        for filename in os.listdir("rag_files/contracts"):
            file_path = f"rag_files/contracts/{filename}"

            if os.path.isdir(file_path):
                continue

            contract_content = load_file_contents(file_path)
            if not contract_content:
                continue

            contract_documents.append(contract_content)
            contract_type_name = filename.replace(".txt", "")

            contract_metadata.append(
                {"source": file_path, "contract_type": contract_type_name}
            )
            contract_ids.append(f"contract_{len(contract_documents)}")

        embeddings = [embed_model.embed_query(doc) for doc in contract_documents]
        contracts_collection.add(
            documents=contract_documents,
            embeddings=embeddings,
            metadatas=contract_metadata,
            ids=contract_ids,
        )
        print(f"Base inicializada com {len(contract_documents)} templates")

    print("Identificando tipo de contrato mais adequado...")
    query_embedding = embed_model.embed_query(question)
    results = contracts_collection.query(
        query_embeddings=[query_embedding],
        n_results=1,
        include=["documents", "metadatas"],
    )

    if not results["documents"] or len(results["documents"][0]) == 0:
        error_response = {
            "status": "error",
            "answer": "Não foi possível identificar o tipo de contrato solicitado. Por favor, seja mais específico.",
            "message": "Não foi possível identificar o tipo de contrato.",
        }
        print("Erro: Tipo de contrato não identificado")
        return json.dumps(error_response, ensure_ascii=False, indent=2)

    best_match_metadata = results["metadatas"][0][0]
    contract_type_name = best_match_metadata.get("contract_type", "")
    contract_template = results["documents"][0][0]
    
    print(f"Tipo identificado: {contract_type_name}")

    contract_info = fields_mapping.get(contract_type_name, {})
    
    print("Validando correspondência com solicitação do usuário...")
    validation = contract_filter(contract_type_name, question, history)
    
    if not validation.get("match", False):
        error_response = {
            "status": "error",
            "answer": f"Sua requisição não corresponde a nenhum dos nossos tipos de contrato padrão. Por favor, verifique e tente novamente.",
            "message": validation.get('reason', 'Tipo de contrato incompatível'),
            "confidence": validation.get('confidence', 'low')
        }
        print(f"Validação falhou: {validation.get('reason', '')}")
        return json.dumps(error_response, ensure_ascii=False, indent=2)
    
    print(f"Validação aprovada (confiança: {validation.get('confidence', 'medium')})")

    print("Preenchendo template do contrato...")
    template = load_prompt("fill_contract")
    prompt = PromptTemplate.from_template(template)
    llm_chain = prompt | llm | StrOutputParser()
    result = llm_chain.invoke(
        {
            "template": contract_template[:3000],
            "required_fields": json.dumps(
                contract_info.get("required_fields", []), ensure_ascii=False, indent=2
            )[:1000],
            "user_data": question,
            "history": history[-500:],
        }
    )

    try:
        result_data = json.loads(result)
        if result_data.get("status") == "completed":
            with open("generated_contract.txt", "w", encoding="utf-8") as f_out:
                f_out.write(result_data.get("filled_contract", ""))
            print("\nContrato gerado com sucesso!\n")
        elif result_data.get("status") == "missing_info":
            print("\nInformações adicionais necessárias\n")

        return json.dumps(result_data, ensure_ascii=False, indent=2)
    except Exception:
        print("\nProcessamento concluído\n")
        return result


@tool
def retrieve_brazilian_law_context_and_answer(question: str, history: str) -> str:
    """
    Realiza busca e geração de resposta (RAG) sobre legislação brasileira.

    Busca trechos relevantes da legislação brasileira usando expansão de query e similaridade vetorial,
    depois gera uma resposta fundamentada com o assistente jurídico ContratAI.

    Args:
        question (str): Pergunta jurídica a ser respondida
        history (str): Histórico da conversa

    Returns:
        str: Resposta jurídica gerada baseada no contexto recuperado das leis brasileiras
    """
    print("\nConsultando legislação brasileira...")
    print(f"Pergunta: {question}")

    laws_collection = client.get_or_create_collection("laws_collection")
    if laws_collection.count() == 0:
        print("Inicializando base de legislação (primeira execução)...")
        law_rag_raw_path = []
        law_rag_raw_content = []

        for i in os.listdir("rag_files/laws"):
            law_rag_raw_path.append(f"rag_files/laws/{i}")
            law_rag_raw_content.append(load_file_contents(law_rag_raw_path[-1]))

        law_text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=200,
            length_function=len,
            separators=[
                "\n\n",
                "\nArtigo. ",
                "\nTÍTULO ",
                "\nCAPÍTULO ",
                "\nSeção ",
                "\n§ ",
                "\nParágrafo único",
                "\nI - ",
                "\nII - ",
                "\nIII - ",
                "\n",
                ". ",
                " ",
                "",
            ],
        )
        metadata_chunks = []
        chunks_formation(
            law_rag_raw_path, law_rag_raw_content, law_text_splitter, metadata_chunks
        )
        collection_embeding(laws_collection, metadata_chunks, "law")
        print(f"Base inicializada com {laws_collection.count()} chunks de legislação\n")

    print("Filtrando leis relevantes...")
    filter_result = law_question_filter(question, history)

    if filter_result:
        print(f"Buscando em: {', '.join(filter_result)}")
    else:
        print("Buscando em toda a base de legislação")

    query_embedding = embed_model.embed_query(question)
    search_params = {
        "query_embeddings": [query_embedding],
        "n_results": 25,
        "include": ["documents", "metadatas", "distances"],
    }

    if filter_result:
        search_params["where"] = {"source": {"$in": filter_result}}

    results = laws_collection.query(**search_params)

    print("Expandindo busca com variações da pergunta...")
    expanded_results = expand_query(question, laws_collection)

    relevant_docs = []
    seen_docs = set()

    for doc in results["documents"][0]:
        doc_hash = hash(doc[:100])
        if doc_hash not in seen_docs:
            relevant_docs.append(doc)
            seen_docs.add(doc_hash)

    for expanded in expanded_results:
        for doc in expanded["documents"][0][:2]:
            doc_hash = hash(doc[:100])
            if doc_hash not in seen_docs and len(relevant_docs) < 25:
                relevant_docs.append(doc)
                seen_docs.add(doc_hash)

    print(f"Encontrados {len(relevant_docs)} trechos relevantes")

    if relevant_docs:
        print("Reordenando por relevância...")
        pairs = [[question, doc] for doc in relevant_docs]
        scores = reranker.predict(pairs)

        doc_score_pairs = list(zip(relevant_docs, scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)

        relevant_docs = [doc for doc, score in doc_score_pairs[:5]]
        print("Selecionados top 5 trechos mais relevantes")

    combined_context = "\n\n".join(relevant_docs)

    print("Gerando resposta fundamentada...")
    template = load_prompt("retrieve_law")
    prompt = PromptTemplate.from_template(template)
    llm_chain = prompt | llm | StrOutputParser()
    generation = llm_chain.invoke(
        {"context": combined_context, "question": question, "history": history}
    )
    print("\nConsulta jurídica concluída!\n")
    return generation
