# RAG_MultiAgent_Langgraph

Multi-Agent LangGraph (Finance vs General Q&A)

This project demonstrates a multi-agent system using LangGraph
, Ollama
, and FAISS
.
It routes user queries between a Finance Agent (with retrieval-augmented generation from finance documents) and a General Agent (direct LLM response).

🚀 Features

Router Agent: Classifies questions as FINANCE or GENERAL.

Finance Agent: Retrieves relevant context from finance_docs.txt using FAISS and answers with LLM.

General Agent: Answers non-financial queries directly via Ollama.

LangGraph workflow: Manages agent routing and execution flow.

Install dependencies
pip install langchain langgraph langchain-ollama faiss-cpu

Install Ollama:
Follow Ollama installation guide
 and ensure a model (e.g., mistral) is available:       ollama pull mistral
