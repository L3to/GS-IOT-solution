import os
import re
import shutil

# Configurações de Pastas
INPUT_DIR = "rag_files/laws"        # Onde estão os TXTs originais (sujos)
OUTPUT_DIR = "rag_files/laws_clean" # Onde ficarão os arquivos limpinhos

def normalize_law_text(content: str) -> str:
    """
    Higieniza o texto bruto do Planalto/Vade Mecum.
    """
    if not content:
        return ""

    # 1. Normaliza quebras de linha (Windows para Linux/Unix)
    text = content.replace('\r\n', '\n')

    # 2. Remove espaços em branco antes e depois de cada linha
    # Isso resolve o problema da indentação maluca (tabs vs espaços)
    lines = text.split('\n')
    cleaned_lines = [line.strip() for line in lines]
    text = '\n'.join(cleaned_lines)

    # 3. Garante que "Art." comece sempre em uma nova linha
    # Regex: Procura "Art." que NÃO esteja precedido por quebra de linha
    text = re.sub(r'(?<!\n)Art\.', '\nArt.', text)
    
    # 4. Remove caracteres de tabulação residuais
    text = text.replace('\t', ' ')

    # 5. Reduz múltiplos espaços em branco para um só
    # Ex: "Lei    Penal" vira "Lei Penal"
    text = re.sub(r' +', ' ', text)

    # 6. Reduz múltiplas quebras de linha para no máximo duas
    # (Mantém parágrafos, mas remove buracos gigantes no texto)
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()

def process_files():
    # Cria a pasta de saída se não existir
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Pasta criada: {OUTPUT_DIR}")
    else:
        # Opcional: Limpa a pasta antes de começar para não misturar versões
        shutil.rmtree(OUTPUT_DIR)
        os.makedirs(OUTPUT_DIR)
        print(f"Pasta limpa: {OUTPUT_DIR}")

    files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".txt")]
    
    print(f"Encontrados {len(files)} arquivos para limpar.\n")

    for filename in files:
        input_path = os.path.join(INPUT_DIR, filename)
        output_path = os.path.join(OUTPUT_DIR, filename)

        try:
            with open(input_path, 'r', encoding='utf-8') as f_in:
                raw_content = f_in.read()

            clean_content = normalize_law_text(raw_content)

            with open(output_path, 'w', encoding='utf-8') as f_out:
                f_out.write(clean_content)

            print(f"✅ Sucesso: {filename}")
            
        except Exception as e:
            print(f"❌ Erro em {filename}: {e}")

    print("\nProcesso concluído! Agora aponte seu RAG para a pasta 'rag_files/laws_clean'.")

if __name__ == "__main__":
    process_files()