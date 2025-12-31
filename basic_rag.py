from pprint import pprint

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter



loader = PyPDFLoader('hr_manual.pdf')

text_splitter = RecursiveCharacterTextSplitter(chunk_size=300,chunk_overlap=50)

text_splits = text_splitter.split_documents(loader.load())

embedding_model = OllamaEmbeddings(model="nomic-embed-text:latest")

vector_store = InMemoryVectorStore(embedding=embedding_model)

vector_store.add_documents(documents=text_splits)#stores the vector embeddings to retrive later

retrieve_documents = vector_store.similarity_search("what is the promotion policy")

llm_model = ChatOllama(model="llama3.2:latest")

def retrieve_context(query:str):
    retrieved_documents = vector_store.similarity_search(query,k=5)

    content = "\n\n".join(
        (f"Source: {doc.metadata} \nContent : {doc.page_content}" for doc in retrieved_documents)
    )
    return content

prompt_template = ChatPromptTemplate.from_messages([
    ("system",""" You are a helpful assistant who provides answers using the provided context.
                  Use only the  information from context to answer.
                  If context doesn't have the answer say so.
    """),
    ("human", "Context: \n{context} \n\n Question: {question}")
])

rag_chain = prompt_template | llm_model

while 1:
    question = input("What is your question ")

    resp = rag_chain.invoke({
        "context":retrieve_context(question),
        "question" : question
    })
    pprint(resp.content)