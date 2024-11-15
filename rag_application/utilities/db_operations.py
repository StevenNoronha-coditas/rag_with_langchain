from langchain_postgres import PGVector
from langchain_postgres.vectorstores import PGVector
from utilities.helper_functions import get_embedding_model, generate_transcriptions
from dotenv import load_dotenv
import os

load_dotenv()

# Returns a PGVector object that is used to store and retrieve data from Vector DB
def establish_connection():
    connection = os.environ.get("DB_CONNECTION_URL")
    collection_name = os.environ.get("COLLECTION_NAME")
    embeddings = get_embedding_model()

    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=collection_name,
        connection=connection,
        use_jsonb=True,
    )
    return vector_store

# Stores the emeddings in DB using the embed model instanciated during vector object creation, uses helper fns to fetch data
def store_embeddings():
    vector_store = establish_connection()
    data = generate_transcriptions()
    vector_store.add_documents(data)

# Simple fetch data using single query, uses vector.similarity_search to search related documents
def get_data(query):
    vector_store = establish_connection()
    docs = vector_store.similarity_search(query=query, k=1)
    return docs[0].page_content

def get_retriever():
    vector_store = establish_connection()
    retriever = vector_store.as_retriever()
    return retriever
