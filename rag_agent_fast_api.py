import os
from pprint import pprint

from dotenv import load_dotenv
from fastapi import FastAPI
from langchain.agents import create_agent
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel

app = FastAPI(title="HR Rang agent")

class QueryRequest(BaseModel):
    question:str

class QueryResponse(BaseModel):
    question:str
    answer:str

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

loader = PyPDFLoader('hr_manual.pdf')

# text_splitter = RecursiveCharacterTextSplitter(chunk_size=300,chunk_overlap=50)
#
# text_splits = text_splitter.split_documents(loader.load())

embedding_model = OllamaEmbeddings(model="nomic-embed-text:latest")

vector_store = PineconeVectorStore(index_name="weekdayhrdocument",embedding=embedding_model)

# vector_store.add_documents(documents=text_splits)#stores the vector embeddings to retrive later

retrieve_documents = vector_store.similarity_search("what is the promotion policy")

llm_model = ChatOllama(model="llama3.2:latest")

@tool
def retrieve_context(query:str):
    """Retrieve information from vector database to help answer user queries"""
    retrieved_documents = vector_store.similarity_search(query,k=5)

    content = "\n\n".join(
        (f"Source: {doc.metadata} \nContent : {doc.page_content}" for doc in retrieved_documents)
    )
    return content
system_prompt = """
You have access to a tool that retrieves context from hr documents.
Use the tool to help answer user queries.
"""

tools = [retrieve_context]
agent = create_agent(model=llm_model,system_prompt=system_prompt,tools=tools)

# while 1:
#     question = input("What is your question ")
#
#     resp = agent.invoke({"messages" : [{"role":"user","content":question}]})
#     print(resp["messages"][-1].content)

@app.post("/query",response_model=QueryResponse)
def query(request:QueryRequest):
    resp = agent.invoke({"messages":[{"role":"user","content": request.question }]})

    return QueryResponse(
        question=request.question,
        answer=resp["messages"][-1].content
    )

