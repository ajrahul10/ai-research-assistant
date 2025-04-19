import os
from dotenv import load_dotenv
from document_loader import load_and_split_document
from tools.search_docs import create_vector_store, query_documents

load_dotenv()

if __name__ == "__main__":
    file_path = "data/portfolio.pdf"
    docs = load_and_split_document(file_path)
    vectorstore = create_vector_store(docs)

    query = input("Ask a question about the document: ")
    answer = query_documents(vectorstore, query)
    print("Answer:", answer)
