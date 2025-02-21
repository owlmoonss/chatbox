import os
import logging
import gradio as gr
from langchain_community.graphs import Neo4jGraph
from langchain_ollama import OllamaLLM
from langchain.chains import GraphCypherQAChain

logging.basicConfig(level=logging.INFO)
logging.info('Starting up the Knowledge Graph RAG...')

# Instantiate the Neo4J connector
logging.info(f'Instantiating the Neo4J connector for: { os.getenv("NEO4J_URI") }')
NEO4J_URI = "neo4j+s://c2862e70.databases.neo4j.io"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "U1LBvdx9vE2hsrCVY3OJAfYS0Kb0_aIkfviFGsRJz_4"
graph = Neo4jGraph(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)

# Instantiate LLM to use with the Graph RAG
logging.info('Instantiating LLM to use with the LLMGraphTransformer')
llm = OllamaLLM(model='deepseek-r1', temperature=0.0)

# Instantiate the langchain Graph RAG with the Neo4J connector and the LLM
chain = GraphCypherQAChain.from_llm(graph=graph, llm=llm, verbose=True, allow_dangerous_requests=True)
logging.info('Knowledge Graph RAG is ready to go!')
logging.info('=' * 50)

def query_graph_rag(question):
    result = chain.invoke({"query": question})
    return result['result'] if result['result'] else "No response."

# Create a Gradio interface
demo = gr.Interface(fn=query_graph_rag, inputs="text", outputs="text", title="Knowledge Graph RAG")

demo.launch()