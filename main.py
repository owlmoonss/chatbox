import re
import gradio as gr
import networkx as nx
import spacy
from concurrent.futures import ThreadPoolExecutor
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_ollama import OllamaEmbeddings
from chromadb.config import Settings
from chromadb import Client
from langchain_community.vectorstores import Chroma

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

# Parallel embedding generation
def generate_embedding(chunk):
    return embedding_function.embed_query(chunk.page_content)

with ThreadPoolExecutor() as executor:
    embeddings = list(executor.map(generate_embedding, chunks))

# Initialize Chroma client
client = Client(Settings())
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

# ================== Knowledge Graph ================== #
knowledge_graph = nx.DiGraph()

def extract_entities(text):
    """Extract entities using spaCy"""
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

def build_knowledge_graph(chunks):
    """Build Knowledge Graph from document chunks"""
    for chunk in chunks:
        entities = extract_entities(chunk.page_content)
        for i in range(len(entities) - 1):
            node1, type1 = entities[i]
            node2, type2 = entities[i + 1]
            if not knowledge_graph.has_edge(node1, node2):
                knowledge_graph.add_edge(node1, node2, relation=f"{type1} → {type2}")

# Build Knowledge Graph
build_knowledge_graph(chunks)

def query_knowledge_graph(question):
    """Check if any entity in the question exists in the graph"""
    entities = extract_entities(question)
    related_contexts = []

    for entity, _ in entities:
        if entity in knowledge_graph:
            neighbors = list(knowledge_graph.neighbors(entity))
            related_contexts.extend(neighbors)

    return "\n".join(related_contexts)

# ================== RAG Pipeline ================== #
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
    title="RAG Chatbot with Knowledge Graph",
    description="Ask questions about Foundations of LLMs. Uses both ChromaDB and a Knowledge Graph."
)
interface.launch()
