# from langchain_community.llms import Ollama
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaEmbeddings

from langchain_community.document_loaders import TextLoader
from langgraph.graph import StateGraph, END

# ---------------------------
# Define state
# ---------------------------
class State(dict):
    question: str
    context: str
    answer: str

# ---------------------------
# Load documents (finance knowledge base)
# ---------------------------
loader = TextLoader("finance_docs.txt")   # your finance file
docs = loader.load()

embeddings = OllamaEmbeddings(model="mistral")   # open-source embeddings
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever()

# ---------------------------
# Use Ollama as LLM
# ---------------------------
# llm = Ollama(model="mistral")   # or "llama2", "gemma", "phi"
llm = OllamaLLM(model="mistral")

# ---------------------------
# Router Agent
# ---------------------------
def router_node(state: State):
    q = state["question"]
    decision_prompt = f"""
    Decide if this question is FINANCE-related or GENERAL.
    Question: {q}
    Reply with only one word: FINANCE or GENERAL.
    """
    decision = llm.invoke(decision_prompt).strip().upper()
    state["route"] = "FINANCE" if "FINANCE" in decision else "GENERAL"
    return state

# ---------------------------
# Finance Agent
# ---------------------------
def finance_node(state: State):
    # docs = retriever.get_relevant_documents(state["question"])
    docs = retriever.invoke(state["question"])

    state["context"] = "\n".join([d.page_content for d in docs])
    prompt = f"Context:\n{state['context']}\n\nQ: {state['question']}\nA:"
    state["answer"] = llm.invoke(prompt)
    return state

# ---------------------------
# General Agent
# ---------------------------
def general_node(state: State):
    state["answer"] = llm.invoke(state["question"])
    return state

# ---------------------------
# Build the graph
# ---------------------------
graph = StateGraph(State)
graph.add_node("router", router_node)
graph.add_node("finance", finance_node)
graph.add_node("general", general_node)

graph.set_entry_point("router")

graph.add_conditional_edges(
    "router",
    lambda s: s["route"],
    {"FINANCE": "finance", "GENERAL": "general"},
)

graph.add_edge("finance", END)
graph.add_edge("general", END)

app = graph.compile()

# ---------------------------
# Test queries
# ---------------------------
# finance_q = {"question": "What are the latest tax benefits for startups?"}
finance_q = {"question":"What is angel tax?"}
finance_ans = app.invoke(finance_q)
print("Finance Answer:", finance_ans["answer"])

general_q = {"question": "Who is the president of India?"}
general_ans = app.invoke(general_q)
print("General Answer:", general_ans["answer"])
