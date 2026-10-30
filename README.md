# RAG Customer Support Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers customer questions for a fictional e-commerce company, **ShopEasy**, using only its policy documents. Every answer comes from the retrieved policy text. If the documents don't cover a question, the bot says so instead of inventing an answer.

Built with **LangChain (LCEL)**, **Groq (Llama 3.3 70B)**, **HuggingFace embeddings**, and **ChromaDB**.

## How it works

```
INDEXING (once)
  Policy doc ──▶ Chunks (1000 chars, 200 overlap) ──▶ Embeddings (MiniLM, 384-d) ──▶ ChromaDB

QUERY (per question)
  Question ──▶ Embedding ──▶ Top-4 similar chunks ──▶ Prompt (context + question) ──▶ Llama 3.3 70B ──▶ Answer
```

| Stage | Tool |
|-------|------|
| Document loading | `TextLoader` / `PyPDFLoader` (Markdown and PDF) |
| Chunking | `RecursiveCharacterTextSplitter` |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (runs locally, free) |
| Vector store | ChromaDB (persisted locally) |
| LLM | Groq `llama-3.3-70b-versatile` |
| Orchestration | LangChain Expression Language (LCEL) |

The core chain:

```python
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | PROMPT
    | llm
    | StrOutputParser()
)
```

The prompt tells the model to answer only from the retrieved context and to fall back to a support-line message when the answer isn't there.

## Example

```
❓ Customer: What is the return policy for electronics?
🤖 ShopEasy Assistant: Electronics can be returned within 10 days of delivery...
```

## Project structure

```
├── RAG_Customer_Support_Chatbot.ipynb   # Full pipeline, step by step, with an interactive chat loop
├── shopeasy_policy.md                   # Knowledge base: terms, privacy, shipping, returns, payments
├── requirements.txt
└── .env.example                         # Template for the Groq API key
```

## Getting started

Requires Python 3.12 and a free [Groq API key](https://console.groq.com/keys).

```bash
git clone https://github.com/manik1848/pdf-rag-chatbot.git
cd pdf-rag-chatbot

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env        # then put your GROQ_API_KEY in .env
jupyter notebook RAG_Customer_Support_Chatbot.ipynb
```

Run the cells from top to bottom. The embedding model (~90 MB) downloads on the first run. The final cell starts an interactive chat; type `quit` to exit.

To use your own PDF instead of the sample policy, swap the loader in Step 2:

```python
loader = PyPDFLoader("your_document.pdf")
```

## Roadmap

- [ ] Source citations in responses
- [ ] Conversation memory for follow-up questions
- [ ] Hybrid search (BM25 + semantic) and re-ranking
- [ ] Streamlit / Gradio web UI
- [ ] Evaluation with RAGAS
