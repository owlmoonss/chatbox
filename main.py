import re
import gradio as gr
import spacy
import networkx as nx
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_ollama import OllamaEmbeddings
from chromadb.config import Settings
from chromadb import Client
from langchain_community.vectorstores import Chroma
from langchain.graphs import Neo4jGraphStore
from langchain.graph_transformers import LLMGraphTransformer

# Load NLP model
nlp = spacy.load("en_core_web_sm")

# Load document
loader = PyMuPDFLoader("2501.09223v1.pdf")
documents = loader.load()

# Split document
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

# Initialize embeddings
embedding_function = OllamaEmbeddings(model="deepseek-r1")

# Initialize ChromaDB
client = Client(Settings())
client.delete_collection(name="foundations_of_llms")
collection = client.create_collection(name="foundations_of_llms")

# Store document embeddings
for idx, chunk in enumerate(chunks):
    embedding = embedding_function.embed_query(chunk.page_content)
    collection.add(
        documents=[chunk.page_content],
        metadatas=[{'id': idx}],
        embeddings=[embedding],
        ids=[str(idx)]
    )

retriever = Chroma(collection_name="foundations_of_llms", client=client, embedding_function=embedding_function).as_retriever()

# ================== Neo4j Knowledge Graph ================== #
neo4j_url = "bolt://localhost:7687"
neo4j_user = "neo4j"
neo4j_password = "password"
neo4j_store = Neo4jGraphStore(url=neo4j_url, username=neo4j_user, password=neo4j_password)

# Use LLMGraphTransformer to extract entities and relationships
graph_transformer = LLMGraphTransformer()
for chunk in chunks:
    graph = graph_transformer.convert(chunk.page_content)
    neo4j_store.add_graph(graph)

def query_knowledge_graph(question):
    """Query Neo4j Knowledge Graph"""
    entities = nlp(question).ents
    query = "MATCH (n)-[r]->(m) WHERE n.name IN $entities RETURN m.name"
    results = neo4j_store.query(query, {"entities": [e.text for e in entities]})
    return "\n".join([res["m.name"] for res in results])

# ================== RAG Pipeline ================== #
def retrieve_context(question):
    kg_context = query_knowledge_graph(question)
    db_results = retriever.invoke(question)
    db_context = "\n\n".join([doc.page_content for doc in db_results])
    return f"KG Context:\n{kg_context}\n\nDB Context:\n{db_context}"

def query_deepseek(question, context):
    formatted_prompt = f"Question: {question}\n\nContext: {context}"
    response = embedding_function.chat(
        model="deepseek-r1",
        messages=[{'role': 'user', 'content': formatted_prompt}]
    )
    response_content = response['message']['content']
    final_answer = re.sub(r'<think>.*?</think>', '', response_content, flags=re.DOTALL).strip()
    return final_answer

def ask_question(question):
    context = retrieve_context(question)
    answer = query_deepseek(question, context)
    return answer

# ================== Gradio Interface ================== #
interface = gr.Interface(
    fn=ask_question,
    inputs="text",
    outputs="text",
    title="RAG Chatbot with Neo4j Knowledge Graph",
    description="Ask questions about Foundations of LLMs. Uses ChromaDB and Neo4j Graph Store."
)
interface.launch()
