import tempfile

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter

def process_pdf(uploaded_file):
    # Save file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    # Use LangChain loader
    loader = PyMuPDFLoader(tmp_file_path)
    documents = loader.load()
    return documents

def create_vector_store(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore
        
def query_vector_store(query, vectorstore):
    retriever = vectorstore.as_retriever()
    relevant_docs = retriever.get_relevant_documents(query)

    llm = ChatOpenAI(temperature=0)
    qa_chain = load_qa_chain(llm, chain_type="stuff")
    result = qa_chain.run(input_documents=relevant_docs, question=query)

    return result
