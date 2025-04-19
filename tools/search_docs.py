from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

def create_vector_store(docs):
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(docs, embeddings)
    return db

def query_documents(vectorstore, query: str):
    retriever = vectorstore.as_retriever()
    llm = ChatOpenAI(temperature=0)
    
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff"  # You can also use "map_reduce", "refine", etc.
    )

    result = qa.invoke({"query": query})
    return result["result"]
