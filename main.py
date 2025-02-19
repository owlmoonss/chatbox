import re
import gradio as gr
import networkx as nx
import spacy
from concurrent.futures import ThreadPoolExecutor
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from chromadb.config import Settings
from chromadb import Client
from langchain_chroma import Chroma
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_neo4j import Neo4jGraph

# Load NLP model
nlp = spacy.load("en_core_web_sm")

# Load text data
text = """Sarah is an employee at prismaticAI, a leading technology company based in Westside Valley. She has been working there for the past three years as a software engineer.
Michael is also an employee at prismaticAI, where he works as a data scientist. He joined the company two years ago after completing his graduate studies.
prismaticAI is a well-known technology company that specializes in developing cutting-edge software solutions and artificial intelligence applications. The company has a diverse workforce of talented individuals from various backgrounds.
Both Sarah and Michael are highly skilled professionals who contribute significantly to prismaticAI's success. They work closely with their respective teams to develop innovative products and services that meet the evolving needs of the company's clients."""

documents = [Document(page_content=text)]

# Split document
text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
chunks = text_splitter.split_documents(documents)

# Initialize embeddings
embedding_function = OllamaEmbeddings(model="deepseek-r1")

# Parallel embedding generation
def generate_embedding(chunk):
    return embedding_function.embed_query(chunk.page_content)

with ThreadPoolExecutor() as executor:
    embeddings = list(executor.map(generate_embedding, chunks))

# Initialize Chroma client
client = Client(Settings())

existing_collections = [col.name for col in client.list_collections()]
if "foundations_of_llms" in existing_collections:
    client.delete_collection(name="foundations_of_llms")

collection = client.create_collection(name="foundations_of_llms")

# Add documents and embeddings
for idx, chunk in enumerate(chunks):
    collection.add(
        documents=[chunk.page_content],
        metadatas=[{'id': idx}],
        embeddings=[embeddings[idx]],
        ids=[str(idx)]
    )

# Initialize retriever
retriever = Chroma(collection_name="foundations_of_llms", client=client, embedding_function=embedding_function).as_retriever()

# ================== Knowledge Graph with Neo4j ================== #
NEO4J_URI = "neo4j+s://c2862e70.databases.neo4j.io"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "U1LBvdx9vE2hsrCVY3OJAfYS0Kb0_aIkfviFGsRJz_4"

neo4j_graph = Neo4jGraph(uri=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)
graph_transformer = LLMGraphTransformer(neo4j_graph)

# Build Knowledge Graph
graph_transformer.transform(chunks)

def query_knowledge_graph(question):
    """Retrieve related entities from Neo4j"""
    entities = extract_entities(question)
    related_contexts = []

    for entity, _ in entities:
        result = neo4j_graph.query(f"MATCH (n)-[r]->(m) WHERE n.name = '{entity}' RETURN m.name")
        related_contexts.extend([record['m.name'] for record in result])

    return "\n".join(related_contexts)

def retrieve_context(question):
    """Retrieve relevant context using both Knowledge Graph & ChromaDB"""
    kg_context = query_knowledge_graph(question)
    db_results = retriever.invoke(question)
    db_context = "\n\n".join([doc.page_content for doc in db_results])
    
    return f"KG Context:\n{kg_context}\n\nDB Context:\n{db_context}"

def query_deepseek(question, context):
    """Query DeepSeek-R1 model"""
    formatted_prompt = f"Question: {question}\n\nContext: {context}"
    response = embedding_function.chat(
        model="deepseek-r1",
        messages=[{'role': 'user', 'content': formatted_prompt}]
    )
    response_content = response['message']['content']
    final_answer = re.sub(r'<think>.*?</think>', '', response_content, flags=re.DOTALL).strip()
    return final_answer

def ask_question(question):
    """Pipeline to retrieve context and generate response"""
    context = retrieve_context(question)
    answer = query_deepseek(question, context)
    return answer

# ================== Gradio Interface ================== #
interface = gr.Interface(
    fn=ask_question,
    inputs="text",
    outputs="text",
    title="RAG Chatbot with Knowledge Graph & Neo4j",
    description="Ask questions about Foundations of LLMs. Uses both ChromaDB and a Knowledge Graph in Neo4j."
)
interface.launch()
