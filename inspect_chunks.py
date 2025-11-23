import chromadb
from chromadb.config import Settings

client = chromadb.PersistentClient(
    path="chroma_db_laws", settings=Settings(anonymized_telemetry=False)
)

collections = client.list_collections()
print(f"Coleções encontradas: {len(collections)}")
for collection in collections:
    print(f"  - {collection.name}")
print()

collection_name = input(
    "Digite o nome da coleção para inspecionar (ou pressione Enter para a primeira): "
).strip()
if not collection_name and collections:
    collection_name = collections[0].name

collection = client.get_collection(name=collection_name)

count = collection.count()
print(f"\nEstatísticas da coleção '{collection_name}':")
print(f"   Total de chunks: {count}")
print()

print("Escolha uma opção:")
print("1. Ver primeiros N chunks")
print("2. Ver chunk específico por ID")
print("3. Buscar chunks por fonte (arquivo)")
print("4. Exportar todos os chunks para arquivo")
print("5. Ver estatísticas por fonte")

choice = input("\nOpção: ").strip()

if choice == "1":
    n = int(input("Quantos chunks mostrar? "))
    results = collection.get(limit=n, include=["documents", "metadatas"])

    print(f"\n{'=' * 80}")
    for i, (doc, meta) in enumerate(zip(results["documents"], results["metadatas"])):
        print(f"\nChunk {i + 1}")
        print(f"   Fonte: {meta.get('source', 'N/A')}")
        print(f"   Chunk Index: {meta.get('chunk_index', 'N/A')}")
        print("   Conteúdo (primeiros 200 chars):")
        print(f"   {doc[:200]}...")
        print(f"\n{'-' * 80}")

elif choice == "2":
    chunk_id = input("Digite o ID do chunk: ").strip()
    results = collection.get(
        ids=[chunk_id], include=["documents", "metadatas", "embeddings"]
    )

    if results["documents"]:
        print(f"\n{'=' * 80}")
        print(f"Chunk ID: {chunk_id}")
        print(f"   Fonte: {results['metadatas'][0].get('source', 'N/A')}")
        print(f"   Chunk Index: {results['metadatas'][0].get('chunk_index', 'N/A')}")
        print(
            f"   Embedding dimension: {len(results['embeddings'][0]) if results['embeddings'] else 'N/A'}"
        )
        print("\n   Conteúdo completo:")
        print(f"   {results['documents'][0]}")
        print(f"\n{'=' * 80}")
    else:
        print(f"Chunk com ID '{chunk_id}' não encontrado")

elif choice == "3":
    source = input("Digite o nome do arquivo fonte: ").strip()
    results = collection.get(
        where={"source": source}, include=["documents", "metadatas"]
    )

    print(f"\nEncontrados {len(results['documents'])} chunks do arquivo '{source}'")
    print(f"\n{'=' * 80}")

    for i, (doc, meta) in enumerate(zip(results["documents"], results["metadatas"])):
        print(f"\nChunk {meta.get('chunk_index', i)}")
        print("   Conteúdo (primeiros 200 chars):")
        print(f"   {doc[:200]}...")
        print(f"\n{'-' * 80}")

        if i >= 9:
            more = len(results["documents"]) - 10
            if more > 0:
                print(f"\n... e mais {more} chunks")
            break

elif choice == "4":
    output_file = input("Nome do arquivo de saída (ex: chunks_export.txt): ").strip()
    results = collection.get(include=["documents", "metadatas"])

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"Exportação de chunks da coleção '{collection_name}'\n")
        f.write(f"Total de chunks: {len(results['documents'])}\n")
        f.write("=" * 80 + "\n\n")

        for i, (doc, meta) in enumerate(
            zip(results["documents"], results["metadatas"])
        ):
            f.write(f"CHUNK {i + 1}\n")
            f.write(f"Fonte: {meta.get('source', 'N/A')}\n")
            f.write(f"Chunk Index: {meta.get('chunk_index', 'N/A')}\n")
            f.write(f"Conteúdo:\n{doc}\n")
            f.write("\n" + "=" * 80 + "\n\n")

    print(f"\n{len(results['documents'])} chunks exportados para '{output_file}'")

elif choice == "5":
    results = collection.get(include=["metadatas"])

    source_counts = {}
    for meta in results["metadatas"]:
        source = meta.get("source", "Unknown")
        source_counts[source] = source_counts.get(source, 0) + 1

    print("\nEstatísticas por fonte:")
    print(f"\n{'Arquivo':<40} {'Chunks':>10}")
    print("-" * 52)
    for source, count in sorted(
        source_counts.items(), key=lambda x: x[1], reverse=True
    ):
        print(f"{source:<40} {count:>10}")
    print("-" * 52)
    print(f"{'TOTAL':<40} {sum(source_counts.values()):>10}")

else:
    print("Opção inválida")

print("\nInspeção concluída!")
