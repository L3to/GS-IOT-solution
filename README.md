# ContratAI - Assistente Jurídico Inteligente

> **FIAP - Global Solution 2024/2025**  
> **Tema:** O Futuro do Trabalho  
> **Curso:** Análise e Desenvolvimento de Sistemas

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-1.0.7-green.svg)](https://www.langchain.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.0.3-orange.svg)](https://github.com/langchain-ai/langgraph)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-red.svg)](https://ollama.ai/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-1.3.4-purple.svg)](https://www.trychroma.com/)

---

## Sobre o Projeto

**ContratAI** é um assistente jurídico inteligente desenvolvido como solução para o desafio **"O Futuro do Trabalho"** da Global Solution FIAP. O sistema utiliza **Inteligência Artificial Generativa** (LLMs locais) combinada com **RAG (Retrieval-Augmented Generation)** para democratizar o acesso à informação jurídica e automatizar processos contratuais complexos.

### Problema Identificado

No futuro do trabalho:
- **Democratização do conhecimento jurídico**: Profissionais autônomos, freelancers e pequenos empreendedores frequentemente não têm acesso a assessoria jurídica acessível
- **Complexidade contratual**: Contratos são documentos técnicos e intimidadores para não-advogados
- **Conformidade legal**: Garantir que contratos estejam em conformidade com a legislação brasileira (LGPD, CC, CLT, CDC)
- **Análise de riscos**: Identificar cláusulas abusivas, omissões e riscos jurídicos sem depender de consultorias caras
- **Automação jurídica**: Reduzir tempo e custos em processos contratuais repetitivos

### Solução Proposta

O ContratAI utiliza **Inteligência Artificial Generativa** (LLMs locais via Ollama) combinada com **RAG (Retrieval-Augmented Generation)** para oferecer:

1. **Análise Profunda de Contratos** - Identificação de riscos jurídicos, cláusulas ausentes, obrigações das partes e conformidade legal com feedback detalhado
2. **Reformulação de Contratos** - Otimização de redação, correção de cláusulas abusivas, adequação à LGPD/CC/CLT/CDC com resumo executivo das mudanças
3. **Consulta à Legislação Brasileira** - Busca semântica inteligente em 8 códigos legais completos com query expansion e reranking
4. **Geração de Contratos Personalizados** - Criação de contratos profissionais baseados em 14 templates especializados com validação de campos
5. **Histórico Persistente** - Armazenamento de conversas no MongoDB Atlas para continuidade e auditoria completa de sessões

---

## Arquitetura do Sistema

### Tecnologias Utilizadas

| Tecnologia | Versão | Função |
|-----------|--------|--------|
| **Python** | 3.11 | Linguagem principal |
| **LangChain** | 1.0.7 | Framework para aplicações LLM |
| **LangGraph** | 1.0.3 | Orquestração de agentes multi-step com grafo de estados |
| **Ollama** | 0.6.1 | Execução local de LLMs |
| **mistral-nemo:12b** | - | Modelo LLM principal (agents e embeddings) |
| **gpt-oss:20b** | - | Modelo para filtragem de arquivos de leis (JSON estruturado) |
| **ChromaDB** | 1.3.4 | Banco vetorial persistente para RAG |
| **MongoDB Atlas** | 4.15.4 | Persistência de histórico de conversas |
| **Sentence Transformers** | 3.0.0 | CrossEncoder reranker (ms-marco-MiniLM-L-6-v2) |
| **PDFPlumber** | 0.11.8 | Extração de texto de PDFs (futuro) |

---

## Arquitetura do Sistema

### Fluxo de Conversa (LangGraph StateGraph)

```mermaid
graph TB
    Start([Usuário envia pergunta]) --> ChatAgent[Chat Agent<br/>Decide: usar tool ou responder direto]
    
    ChatAgent -->|use_tool: false| DirectResponse[Resposta Direta<br/>Conhecimento geral do LLM]
    ChatAgent -->|use_tool: true| ToolAgent[Tool Agent<br/>Seleciona ferramenta e extrai argumentos]
    
    DirectResponse --> End([Resposta ao usuário])
    
    ToolAgent --> ToolExecutor{Tool Executor<br/>Executa ferramenta selecionada}
    
    ToolExecutor -->|analyze_contract| T1[Análise de Contrato<br/>Riscos + Cláusulas + Obrigações]
    ToolExecutor -->|refactor_contract| T2[Reformulação<br/>LGPD + CDC + CC + CLT]
    ToolExecutor -->|retrieve_law| T3[Consulta Legislação<br/>RAG em 8 códigos legais]
    ToolExecutor -->|generate_contracts| T4[Geração de Contratos<br/>Templates profissionais]
    ToolExecutor -->|leave_chat| T5[Encerrar Sessão]
    
    T1 --> SaveMongo[(MongoDB Atlas<br/>Salva histórico)]
    T2 --> SaveMongo
    T3 --> SaveMongo
    T4 --> SaveMongo
    T5 --> SaveMongo
    
    SaveMongo --> ChatAgent
    
    style ChatAgent fill:#4A90E2,stroke:#2E5C8A,color:#fff
    style ToolAgent fill:#7B68EE,stroke:#4B0082,color:#fff
    style ToolExecutor fill:#50C878,stroke:#2E7D4E,color:#fff
    style T3 fill:#FF6B6B,stroke:#C92A2A,color:#fff
    style SaveMongo fill:#FFA500,stroke:#CC8400,color:#fff
```

### Arquitetura RAG + LLM

```mermaid
graph LR
    subgraph Input
        User[Usuário]
    end
    
    subgraph LangGraph
        CA[Chat Agent<br/>mistral-nemo:12b]
        TA[Tool Agent<br/>mistral-nemo:12b]
    end
    
    subgraph RAG_System[Sistema RAG - ChromaDB]
        LC[Laws Collection<br/>8 códigos legais<br/>~2.5M tokens]
        CC[Contracts Collection<br/>14 contratos completos<br/>1 arquivo = 1 chunk]
    end
    
    subgraph Tools
        T1[analyze_contract]
        T2[refactor_contract]
        T3[retrieve_law]
        T4[generate_contracts]
    end
    
    subgraph Persistence
        Mongo[(MongoDB Atlas<br/>Histórico de Sessões)]
    end
    
    User -->|Pergunta| CA
    CA -->|use_tool: true| TA
    TA --> T1 & T2 & T3 & T4
    T3 -->|Query Expansion| LC
    T4 -->|Busca Template| CC
    T1 & T2 & T3 & T4 -->|Resultado| Mongo
    Mongo -->|Contexto| CA
    CA -->|Resposta| User
    
    style LC fill:#FF6B6B,stroke:#C92A2A,color:#fff
    style CC fill:#4ECDC4,stroke:#2A9D8F,color:#fff
    style CA fill:#4A90E2,stroke:#2E5C8A,color:#fff
    style TA fill:#7B68EE,stroke:#4B0082,color:#fff
    style Mongo fill:#FFA500,stroke:#CC8400,color:#fff
```

### Decisão do Chat Agent

```mermaid
flowchart TD
    Question[Pergunta do Usuário] --> CheckLaw{Pergunta sobre<br/>LEIS/ARTIGOS/CÓDIGOS?}
    
    CheckLaw -->|SIM| UseTool[use_tool: true<br/>tools_list: retrieve_brazilian_law]
    CheckLaw -->|NÃO| CheckContract{Menciona caminho<br/>de arquivo de contrato?}
    
    CheckContract -->|SIM| CheckAction{Qual ação?}
    CheckAction -->|Analisar| UseAnalyze[use_tool: true<br/>tools_list: analyze_contract]
    CheckAction -->|Reformular| UseRefactor[use_tool: true<br/>tools_list: refactor_contract]
    
    CheckContract -->|NÃO| CheckGenerate{Pede para<br/>gerar contrato?}
    CheckGenerate -->|SIM| UseGenerate[use_tool: true<br/>tools_list: generate_contracts]
    CheckGenerate -->|NÃO| CheckExit{Quer sair?}
    
    CheckExit -->|SIM| UseExit[use_tool: true<br/>tools_list: leave_chat]
    CheckExit -->|NÃO| DirectAnswer[use_tool: false<br/>Resposta direta do LLM]
    
    UseTool --> Return[Retorna JSON]
    UseAnalyze --> Return
    UseRefactor --> Return
    UseGenerate --> Return
    UseExit --> Return
    DirectAnswer --> Return
    
    style UseTool fill:#50C878,stroke:#2E7D4E,color:#fff
    style UseAnalyze fill:#50C878,stroke:#2E7D4E,color:#fff
    style UseRefactor fill:#50C878,stroke:#2E7D4E,color:#fff
    style UseGenerate fill:#50C878,stroke:#2E7D4E,color:#fff
    style UseExit fill:#50C878,stroke:#2E7D4E,color:#fff
    style DirectAnswer fill:#FFD700,stroke:#FFA500,color:#000
```

---

### Sistema RAG (Retrieval-Augmented Generation)

#### 1. **RAG de Legislação** (laws_collection)
- **Base de dados**: 8 códigos legais brasileiros (~2.5M tokens)
  - Constituição Federal 1988
  - Código Civil (Lei 10.406/2002)
  - Código de Processo Civil
  - Código Penal
  - Código de Processo Penal
  - CLT (Consolidação das Leis do Trabalho)
  - CDC (Código de Defesa do Consumidor)
  - ECA (Estatuto da Criança e do Adolescente)
- **Chunking**: `RecursiveCharacterTextSplitter` com separadores específicos (`Art.`, `TÍTULO`, `CAPÍTULO`)
- **Estratégia**: Query expansion + busca por similaridade vetorial
- **Chunk size**: 1000 chars, overlap: 200

#### 2. **RAG de Contratos** (contracts_collection)
- **Base de dados**: 14 contratos profissionais completos
  - Trabalho CLT
  - Compra/Venda Veículo
  - Parceria Comercial e Distribuição
  - Doação de Imóvel
  - Sociedade Empresária Ltda
  - Cessão de Direitos de Imagem
  - Desenvolvimento de Software
  - Cessão de Quotas Sociais (M&A)
  - Comodato de Imóvel
  - Constituição de Holding Familiar
  - Corretagem Imobiliária
  - Locação Imóvel Urbano
  - Prestação de Serviços
  - Promessa de Compra e Venda
- **Estratégia**: **1 arquivo = 1 chunk** (contexto integral)
- **Vantagem**: Mantém integridade do contrato para análise holística e geração de templates completos
- **Uso**: Template matching para geração de contratos personalizados via busca semântica

---

## Funcionalidades Principais

### 1. Análise de Contratos (`analyze_contract`)

**Análise jurídica profunda com feedback detalhado ao usuário:**

**Logging durante execução:**
- Analisando contrato...
- Carregando documento do caminho especificado...
- Executando análise jurídica profunda...
- Análise concluída com sucesso!

**Saída JSON estruturada:**

```json
{
  "answer": "Análise concluída: Risco MÉDIO identificado. Recomendado adicionar cláusulas de LGPD e revisar cláusula de exclusividade.",
  "metadata": {
    "contract_type": "Prestação de Serviços",
    "parties": {"contractor": "...", "contracted": "..."},
    "date": "01/12/2024",
    "value": "R$ 50.000,00",
    "duration": "12 meses"
  },
  "risk_analysis": {
    "overall_risk": "Médio",
    "high_risks": [
      {
        "description": "Cláusula de exclusividade sem contrapartida",
        "legal_basis": "Art. 422 do Código Civil",
        "impact": "Abusividade contratual",
        "recommendation": "Adicionar cláusula de exclusividade recíproca"
      }
    ]
  },
  "missing_clauses": [
    {"clause_name": "Proteção de Dados (LGPD)", "importance": "Crítica"}
  ],
  "obligations": {
    "contractor": [...],
    "contracted": [...]
  },
  "executive_summary": {
    "rationale": "3-5 parágrafos detalhados com citações legais...",
    "recommendation": "..."
  }
}
```

### 2. Reformulação de Contratos (`refactor_contract`)

**Otimização de contratos com feedback detalhado:**

**Logging durante execução:**
- Iniciando reformulação do contrato...
- Carregando contrato original...
- Executando análise e reformulação...
- Reformulação concluída! Contrato salvo em: refactored_contract.txt
- Tamanho do arquivo: X KB

**Saída JSON estruturada:**

```json
{
  "answer": "Contrato reformulado com sucesso! Principais mudanças: adicionadas 3 cláusulas (LGPD, Força Maior, Não-concorrência), corrigidas 2 cláusulas abusivas. Arquivo salvo em refactored_contract.txt.",
  "refactored_contract": {
    "title": "CONTRATO DE PRESTAÇÃO DE SERVIÇOS - VERSÃO REFORMULADA",
    "summary": "Resumo executivo em 2-3 parágrafos: (1) o que foi reformulado e por quê, (2) principais melhorias implementadas, (3) impacto prático das mudanças",
    "full_text": "Contrato completo reformulado...",
    "structure": [...]
  },
  "changes_made": {
    "additions": ["Cláusula de LGPD", "Cláusula de Força Maior"],
    "modifications": ["Cláusula de rescisão - adequada ao CDC"],
    "removals": ["Cláusula abusiva de renúncia de direitos"]
  },
  "legal_analysis": [...],
  "recommendations": [...],
  "metadata": {...}
}
```

### 3. Consulta à Legislação (`retrieve_brazilian_law_context_and_answer`)

**Busca inteligente em legislação brasileira com pipeline completo:**

**Logging durante execução:**
- Filtrando arquivos de lei relevantes...
- Arquivos selecionados: CONSOLIDACAODASLEISDOTRABALHO.txt, CONSTITUICAOFEDERAL.txt
- Buscando na base de legislação (laws_collection)...
- 15 chunks encontrados
- Expandindo query para melhorar resultados...
- Reordenando resultados por relevância (reranking)...
- Top 5 documentos mais relevantes selecionados

**Pipeline RAG:**
1. **Filtragem de arquivos** - Usa gpt-oss:20b para identificar leis relevantes à pergunta
2. **Query Expansion** - Gera 2 variações da query usando prompt expand_query.txt
3. **Busca Vetorial** - ChromaDB com embeddings mistral-nemo:12b
4. **Reranking** - CrossEncoder (ms-marco-MiniLM-L-6-v2) reordena por relevância
5. **Top-K Selection** - Seleciona 5 documentos mais relevantes

**Exemplo:**
```python
"Quais são os direitos trabalhistas em caso de demissão sem justa causa?"

# Resposta com citações legais
{
  "answer": "Em caso de demissão sem justa causa, o trabalhador tem direito a: saldo de salário, aviso prévio (indenizado ou trabalhado), férias vencidas + 1/3, férias proporcionais + 1/3, 13º salário proporcional, saque do FGTS + multa de 40%, seguro-desemprego. Base legal: Art. 477 da CLT, Art. 7º CF/88.",
  "sources": [
    {"file": "CONSOLIDACAODASLEISDOTRABALHO.txt", "content": "Art. 477..."},
    {"file": "CONSTITUICAOFEDERAL.txt", "content": "Art. 7º..."}
  ]
}
```

### 4. Geração de Contratos (`generate_contracts`)

**Criação de contratos personalizados baseados em 14 templates profissionais:**

**Logging durante execução:**
- Iniciando geração de contrato...
- Buscando template mais adequado na base de contratos...
- Template encontrado: CONTRATO DE PRESTAÇÃO DE SERVIÇOS
- Preenchendo campos do contrato...
- Contrato gerado com sucesso!

**Fluxo de geração:**
1. **Busca de Template** - Identifica contrato similar via busca semântica na contracts_collection
2. **Validação de Campos** - Verifica se todos os campos necessários foram fornecidos
3. **Preenchimento** - Usa prompt fill_contract.txt para preencher template
4. **Validação Legal** - Garante conformidade com LGPD, CC, CLT, CDC

**Exemplo:**
```python
"Crie um contrato de prestação de serviços para desenvolvimento de software entre TechCorp e ClienteSA, valor R$ 50.000, prazo 6 meses"

# Saída
{
  "answer": "Contrato de Prestação de Serviços gerado com sucesso! Baseado no template profissional com cláusulas de LGPD, propriedade intelectual e não-concorrência.",
  "status": "completed",
  "filled_contract": "CONTRATO DE PRESTAÇÃO DE SERVIÇOS\n\nCONTRATANTE: TechCorp...\nCONTRATADA: ClienteSA...\nVALOR: R$ 50.000,00...\n[Contrato completo com todas as cláusulas]"
}

# Caso faltem informações
{
  "answer": "Para gerar o contrato, preciso das seguintes informações adicionais: endereço completo das partes, CNPJ/CPF, forma de pagamento.",
  "status": "missing_info",
  "missing_fields": ["endereco_contratante", "cnpj_contratante", "forma_pagamento"]
}

---

## Estrutura do Projeto

```
ContratAIIOT/
├── main.py                           # Aplicação principal (LangGraph workflow)
├── tools.py                          # Ferramentas/Tools para agentes (5 tools)
├── llm_config.py                     # Configuração Ollama + ChromaDB
├── rag_functions.py                  # Funções RAG (chunking, embedding, query expansion, reranking)
├── contract_fields_mapping.json      # Mapeamento de campos para geração de contratos
├── inspect_chunks.py                 # Utilitário para inspecionar chunks no ChromaDB
├── inspect_sqlite.py                 # Utilitário para inspecionar banco SQLite do ChromaDB
├── .env                              # Variáveis de ambiente (MongoDB, Ollama)
├── pyproject.toml                    # Dependências do projeto (uv)
├── uv.lock                           # Lock file do uv
│
├── prompts/                          # Sistema de prompts externalizados
│   ├── chat_agent.txt                # Prompt do Chat Agent (decisor de routing)
│   ├── tool_agent.txt                # Prompt do Tool Agent (seleção e execução)
│   ├── analyze_contract.txt          # Prompt de análise de contratos (rationale detalhado)
│   ├── refactor_contract.txt         # Prompt de reformulação (com campo summary)
│   ├── retrieve_law.txt              # Prompt de consulta legislação
│   ├── expand_query.txt              # Prompt de expansão de queries para RAG
│   ├── fill_contract.txt             # Prompt de preenchimento de templates
│   └── filter_law_files.txt          # Prompt de filtragem de arquivos de leis
│
├── rag_files/
│   ├── contracts/                    # 14 contratos templates (.txt)
│   │   ├── CONTRATO DE TRABALHO COM REGISTRO CLT.txt
│   │   ├── CONTRATO DE DESENVOLVIMENTO DE SOFTWARE.txt
│   │   ├── CONTRATO DE CONSTITUIÇÃO DE HOLDING FAMILIAR.txt
│   │   ├── CONTRATO DE PRESTAÇÃO DE SERVIÇOS.txt
│   │   ├── CONTRATO DE LOCAÇÃO DE IMÓVEL URBANO.txt
│   │   ├── CONTRATO DE CESSÃO DE DIREITOS DE IMAGEM.txt
│   │   ├── CONTRATO DE PARCERIA COMERCIAL E DISTRIBUIÇÃO.txt
│   │   ├── CONTRATO DE COMPRA E VENDA DE VEÍCULO.txt
│   │   ├── CONTRATO MODELO REAL.txt             # Exemplo de excelência profissional
│   │   ├── CONTRATO DE EXEMPLO COM ERROS.txt    # Exemplo para testes
│   │   └── ... (mais 4 contratos)
│   │
│   └── laws/                         # 8 códigos legais brasileiros (.txt)
│       ├── CONSTITUICAOFEDERAL.txt
│       ├── CODIGOCIVIL.txt
│       ├── CONSOLIDACAODASLEISDOTRABALHO.txt
│       ├── CODIGODEFESADOCONSUMIDOR.txt
│       ├── CODIGOPROCESSOCIVIL.txt
│       ├── CODIGOPENAL.txt
│       ├── CODIGOPROCESSOPENAL.txt
│       └── ESTATUTOCRIANCAADOLESCENTE.txt
│
├── chroma_db_laws/                   # ChromaDB persistente (embeddings)
│   ├── chroma.sqlite3                # Banco SQLite com metadados
│   └── 5dbf0b49-3196-4f82-b5c0-9f603523f1a7/  # Collection data
│
├── __pycache__/                      # Cache Python
└── README.md                         # Este arquivo
```

---

## Instalação e Configuração

### Pré-requisitos

- **Python 3.11**
- **Ollama** instalado ([ollama.ai](https://ollama.ai/))
- **Modelo Ollama**: `mistral-nemo:12b`
- **MongoDB Atlas** (ou local)

### 1. Clone o Repositório

```bash
git clone <repository-url>
cd ContratAIIOT
```

### 2. Instale os Modelos Ollama

```bash
ollama pull mistral-nemo:12b  # Modelo principal (agents e embeddings)
ollama pull gpt-oss:20b       # Modelo para filtragem de leis (JSON estruturado)
```

### 3. Configure as Dependências

**Opção A: Usando `uv` (recomendado)**

```bash
pip install uv
uv sync
```

**Opção B: Usando `pip`**

```bash
pip install -r requirements.txt
# ou manualmente:
pip install langchain langchain-core langchain-community langchain-ollama
pip install langgraph ollama chromadb pymongo python-dotenv
```

### 4. Configure o `.env`

Crie um arquivo `.env` na raiz do projeto:

```env
# MongoDB Atlas
DB_PASSWORD=sua_senha_mongodb

# Ollama (se necessário)
OLLAMA_HOST=http://localhost:11434

# ChromaDB (opcional)
CHROMA_PERSIST_DIRECTORY=./chroma_db_laws
```

### 5. Inicialize o ChromaDB (Primeira Execução)

Na primeira execução, o sistema automaticamente:
- Carrega todos os arquivos de `rag_files/laws/` (8 códigos legais)
- Carrega todos os arquivos de `rag_files/contracts/` (14 templates)
- Cria chunks com `RecursiveCharacterTextSplitter` (tamanho: 1200, overlap: 200)
- Gera embeddings com `OllamaEmbeddings` (mistral-nemo:12b)
- Persiste no ChromaDB (`chroma_db_laws/`)
- Exibe progresso: "Progress: X/Y chunks (Z%)"

**Importante:** 
- Este processo pode levar 5-10 minutos na primeira vez
- Contratos usam estratégia **1 arquivo = 1 chunk** para contexto integral
- Leis são divididas em chunks de 1200 caracteres com separadores (`Art.`, `§`, `CAPÍTULO`)

### 6. Execute o Assistente

```bash
python main.py
```

**Ou ative o ambiente virtual primeiro:**

```bash
source .venv/bin/activate  # Linux/Mac
# ou
.venv\Scripts\activate     # Windows

python main.py
```

---

## Como Usar

### Iniciando uma Sessão

```
Digite o ID da sessão (ou Enter para nova sessão): 

Nova sessão criada com ID: abc12345

Digite sua pergunta jurídica: _
```

### Exemplos de Perguntas

#### 1. Análise de Contrato

```
Analise o contrato em rag_files/contracts/CONTRATO DE TRABALHO COM REGISTRO CLT.txt
```

**O sistema vai:**
- Ler o arquivo
- Identificar metadados (partes, valores, prazos)
- Analisar riscos jurídicos
- Identificar cláusulas ausentes
- Mapear obrigações das partes
- Gerar resumo executivo

#### 2. Consulta à Legislação

```
Quais são os prazos para aviso prévio na CLT?
```

**O sistema vai:**
- Buscar no RAG de legislação (CLT)
- Retornar artigos relevantes (Art. 487, Art. 488)
- Explicar em linguagem acessível
- Fornecer confidence score

#### 3. Reformulação de Contrato

```
Reformule o contrato em rag_files/contracts/CONTRATO DE PRESTAÇÃO DE SERVIÇOS.txt 
adicionando cláusula de LGPD e corrigindo cláusulas abusivas
```

**O sistema vai:**
- Ler o contrato original
- Identificar problemas legais
- Adicionar cláusula de proteção de dados (LGPD)
- Corrigir cláusulas abusivas/ilegais
- Retornar contrato reformulado completo

#### 4. Perguntas Simples (Sem Tool)

```
O que é um contrato de comodato?
```

**O Chat Agent responde diretamente sem chamar tools.**

---

## Testes e Validação

### Limpeza do ChromaDB

Se precisar resetar o banco vetorial:

```bash
python clean_database.py
```

### Sessões Persistentes

O sistema mantém histórico completo no MongoDB Atlas:

```python
# Carregar sessão anterior
Digite o ID da sessão (ou Enter para nova sessão): abc12345

Sessão 'abc12345' carregada! (12 mensagens)

# Estrutura no MongoDB
{
  "session_id": "abc12345",
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."},
    {"role": "tool", "tool_call_id": "...", "name": "analyze_contract", "content": "{...}"}
  ]
}
```

### Ferramentas de Inspeção

**Inspecionar chunks no ChromaDB:**
```bash
python inspect_chunks.py
```

**Inspecionar banco SQLite do ChromaDB:**
```bash
python inspect_sqlite.py
```

---

## Diferenciais do Projeto

### 1. **LLM 100% Local (Privacidade)**
- Nenhum dado enviado para APIs externas (OpenAI, Anthropic)
- Conformidade com LGPD para dados sensíveis
- Zero custo de API

### 2. **RAG Especializado em Legislação Brasileira**
- 8 códigos legais completos (~2.5M tokens)
- Query expansion para melhor recall
- Separadores customizados para estrutura legal (Art., §, incisos)

### 3. **Arquitetura de Agentes (LangGraph)**
- Chat Agent decide quando usar ferramentas
- Tool Agent executa ferramentas específicas
- Separação clara de responsabilidades

### 4. **Contratos Profissionais Completos**
- 14 templates jurídicos detalhados
- Citações legais completas (CF, CC, CLT, CDC, LGPD)
- Cláusulas modernas (LGPD, não-concorrência, governança)

### 5. **Histórico Persistente (MongoDB Atlas)**
- Sessões recuperáveis por ID único
- Auditoria completa de conversas (user/assistant/tool messages)
- Contexto preservado entre execuções
- Suporte a continuação de análises complexas

### 6. **Logging Detalhado**
- Feedback em tempo real de todas as operações
- Progresso de embedding: "Progress: X/Y chunks (Z%)"
- Etapas RAG: filtragem → busca → expansão → reranking
- Confirmação de salvamento de contratos reformulados
- Mensagens user-friendly sem emojis

---

## 🔮 Futuro do Trabalho - Impacto Social

### Como o ContratAI se alinha ao tema "Futuro do Trabalho"?

#### 1. **Democratização do Acesso Jurídico**
- 🎯 **Problema**: Profissionais autônomos e freelancers não têm acesso a assessoria jurídica acessível
- ✅ **Solução**: ContratAI oferece análise jurídica profissional gratuitamente via LLMs locais

#### 2. **Empoderamento de Empreendedores**
- 🎯 **Problema**: Pequenas empresas gastam muito com advogados para contratos simples
- ✅ **Solução**: Geração e análise automatizada de contratos comuns (prestação de serviços, locação, compra/venda)

#### 3. **Conformidade Legal Automatizada**
- 🎯 **Problema**: Contratos desatualizados sem adequação à LGPD e legislação moderna
- ✅ **Solução**: Sistema identifica e corrige automaticamente cláusulas não conformes

#### 4. **Educação Jurídica Acessível**
- 🎯 **Problema**: Linguagem jurídica é técnica e intimidadora para leigos
- ✅ **Solução**: Explicações em linguagem simples + citações legais precisas

#### 5. **Trabalho Remoto e Freelancing**
- 🎯 **Problema**: Crescimento de trabalho remoto aumenta necessidade de contratos claros
- ✅ **Solução**: Templates profissionais para contratos de prestação de serviços, confidencialidade, cessão de direitos

---

## 🛠️ Roadmap / Melhorias Futuras

### Concluído ✅
- [x] **Sistema de prompts externalizados** (8 prompts em arquivos .txt)
- [x] **Logging detalhado** (feedback user-friendly em todas as operações)
- [x] **Geração de contratos personalizados** (função `generate_contracts` completa)
- [x] **Query expansion** (2 variações de query para melhor recall)
- [x] **Reranking com CrossEncoder** (ms-marco-MiniLM-L-6-v2)
- [x] **Filtragem inteligente de leis** (gpt-oss:20b seleciona arquivos relevantes)
- [x] **Resumo executivo em reformulações** (campo summary com 2-3 parágrafos)
- [x] **Rationale detalhado em análises** (3-5 parágrafos com citações legais)

### Em Desenvolvimento 🚧
- [ ] **Hybrid Search** (BM25 + semantic para busca de artigos específicos)
- [ ] **Metadados de artigos** (extração de números de artigos durante chunking)
- [ ] **Interface Web (Streamlit/Gradio)**
- [ ] **Upload de PDFs** (análise de contratos em PDF via `pdfplumber`)

### Planejado 📋
- [ ] **Comparação de contratos** (detectar alterações entre versões)
- [ ] **Assinatura digital** (integração com certificados digitais ICP-Brasil)
- [ ] **Multi-tenancy** (suporte a múltiplas organizações)
- [ ] **API REST** (exposição das funcionalidades via FastAPI)
- [ ] **Fine-tuning do LLM** (especialização em jurídico brasileiro)
- [ ] **Suporte a mais legislações** (Lei de Software, Marco Civil da Internet, Lei de Franquias)
- [ ] **Cache de embeddings** (evitar reprocessamento de contratos já analisados)
- [ ] **Exportação em PDF/DOCX** (contratos reformulados em formatos profissionais)

---

## 👥 Equipe

**FIAP - Global Solution 2024/2025**  
**Tema:** O Futuro do Trabalho

- **Desenvolvedor:** [Seu Nome]
- **RM:** [Seu RM]
- **Curso:** Análise e Desenvolvimento de Sistemas / Engenharia de Software
- **Turma:** [Sua Turma]

---

## 📄 Licença

Este projeto foi desenvolvido para fins acadêmicos como parte do programa Global Solution da FIAP.

---

## 🤝 Contribuições

Contribuições são bem-vindas! Por favor:
1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/NovaFuncionalidade`)
3. Commit suas mudanças (`git commit -m 'Adiciona nova funcionalidade'`)
4. Push para a branch (`git push origin feature/NovaFuncionalidade`)
5. Abra um Pull Request

---

## 📞 Contato

Para dúvidas ou sugestões sobre o projeto:
- **GitHub Issues:** [Abrir issue](../../issues)
- **Email:** [seu-email@fiap.com.br]

---

## Agradecimentos

- **FIAP** - Pela proposta desafiadora da Global Solution
- **LangChain** - Framework poderoso para aplicações LLM
- **Ollama** - Execução local de LLMs de forma simples
- **Comunidade Open Source** - Pelas ferramentas incríveis disponibilizadas

---

<div align="center">

**Democratizando o acesso à justiça através da Inteligência Artificial**

**FIAP Global Solution 2024/2025 - O Futuro do Trabalho**

</div>
