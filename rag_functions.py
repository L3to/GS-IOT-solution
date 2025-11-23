import json

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaEmbeddings

from llm_config import llm

embed_model = OllamaEmbeddings(model="nomic-embed-text")


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
                    "source": file_path,
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
    expansion_template = f"Gere 2 variantes da seguinte pergunta para melhorar a recuperação de informações, o output deve ser em formato JSON com o campo 'variantes' contendo uma lista das perguntas geradas ('variations': ['variation 1', 'variation 2']): {question}"
    expansion_prompt = PromptTemplate.from_template(expansion_template)
    expansion_llm_chain = expansion_prompt | llm | StrOutputParser()
    expansions = expansion_llm_chain.invoke({"question": question})
    data = json.loads(expansions)
    variations = data.get("variations", [])
    expanded_results = []
    for var in variations:
        query_embedding = embed_model.embed_query(var)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3,
        )
        expanded_results.append((results))
    return expanded_results
