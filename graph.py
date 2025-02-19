from langchain_ollama import OllamaLLM
from langchain_experimental.graph_transformers.llm import LLMGraphTransformer
import os
from langchain_community.document_loaders import TextLoader
from langchain.docstore.document import Document  # Create Document from raw text
from langchain_text_splitters import CharacterTextSplitter
import gradio as gr  # Import gradio for the UI

# --- Load and split text data ---
text = """Sarah is an employee at prismaticAI, a leading technology company based in Westside Valley. She has been working there for the past three years as a software engineer.
Michael is also an employee at prismaticAI, where he works as a data scientist. He joined the company two years ago after completing his graduate studies.
prismaticAI is a well-known technology company that specializes in developing cutting-edge software solutions and artificial intelligence applications. The company has a diverse workforce of talented individuals from various backgrounds.
Both Sarah and Michael are highly skilled professionals who contribute significantly to prismaticAI's success. They work closely with their respective teams to develop innovative products and services that meet the evolving needs of the company's clients."""

documents = [Document(page_content=text)]

# Split text into chunks with some overlap
# オーバーラップを含めてテキストをチャンクに分割します。
text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
texts = text_splitter.split_documents(documents)

# --- Initialize LLM using Ollama with deepseek r1 ---
# 深層探索モデル deepseek r1 を使用します
llm = OllamaLLM(model="deepseek-r1", temperature=0)

# --- Extract Knowledge Graph ---
llm_transformer = LLMGraphTransformer(llm=llm)
graph_documents = llm_transformer.convert_to_graph_documents(texts)

from langchain.graph_stores import Neo4jGraphStore

# Kết nối với Neo4j
NEO4J_URI = "neo4j+s://c2862e70.databases.neo4j.io"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "U1LBvdx9vE2hsrCVY3OJAfYS0Kb0_aIkfviFGsRJz_4"

graph_store = Neo4jGraphStore(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)
graph_store.write_graph(graph_documents)

from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import KnowledgeGraphRAGRetriever
from llama_index.core.response_synthesis import ResponseSynthesizer

# --- Initialize ResponseSynthesizer ---
response_synthesizer = ResponseSynthesizer(llm)

# --- Setup the Retriever for Knowledge Graph RAG ---
graph_rag_retriever = KnowledgeGraphRAGRetriever(storage_context=graph_store.storage_context, verbose=True)
query_engine = RetrieverQueryEngine.from_args(graph_rag_retriever)

# --- Define query function ---
def query_and_synthesize(query: str) -> str:
    """
    Retrieve context from the knowledge graph and synthesize an answer.
    グラフからコンテキストを取得し、回答を合成します。
    """
    retrieved_context = query_engine.query(query)
    response = response_synthesizer.synthesize(query, retrieved_context)
    return response

# --- Create Gradio Interface ---
interface = gr.Interface(
    fn=query_and_synthesize,
    inputs=gr.Textbox(lines=2, placeholder="Enter your query here..."),
    outputs="text",
    title="Knowledge Graph Query Interface",
    description="Query the knowledge graph using the deepseek r1 model via Ollama."
)

if __name__ == "__main__":
    interface.launch()
