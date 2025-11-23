import json
import sqlite3

db_path = "chroma_db_laws/chroma.sqlite3"

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

print("Tabelas no banco:")
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
for table in tables:
    print(f"  - {table[0]}")

print("\n" + "=" * 80)

cursor.execute("SELECT * FROM collections;")
collections = cursor.fetchall()
print(f"\nColeções ({len(collections)}):")
for col in collections:
    print(f"  ID: {col[0]} | Nome: {col[1]}")

cursor.execute("""
    SELECT id, document, metadata 
    FROM embeddings 
    LIMIT 10;
""")

print("\n" + "=" * 80)
print("\nPrimeiros 10 chunks:")
rows = cursor.fetchall()
for i, row in enumerate(rows):
    doc_id, document, metadata = row
    meta_dict = json.loads(metadata) if metadata else {}
    print(f"\n[{i + 1}] ID: {doc_id}")
    print(f"    Fonte: {meta_dict.get('source', 'N/A')}")
    print(f"    Preview: {document[:150]}...")

cursor.execute("SELECT COUNT(*) FROM embeddings;")
total = cursor.fetchone()[0]
print(f"\n{'=' * 80}")
print(f"\nTotal de chunks no banco: {total}")

conn.close()
