from langchain_postgres import PGVector
from langchain_postgres.vectorstores import PGVector
from utilities.helper_functions import get_embedding_model, generate_audio_transcriptions, generate_pdf_transcriptions, generate_wiki_transcriptions, generate_youtube_transcriptions
from dotenv import load_dotenv
import os

load_dotenv()

# Returns a PGVector object that is used to store and retrieve data from Vector DB
def establish_connection(collection_name):
    connection = os.environ.get("DB_CONNECTION_URL")
    embeddings = get_embedding_model()

    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=collection_name,
        connection=connection,
        use_jsonb=True,
    )
    return vector_store

# Stores the emeddings in DB using the embed model instanciated during vector object creation, uses helper fns to fetch data
def store_embeddings(collection_name, type_of_data):
    vector_store = establish_connection(collection_name)
    if type_of_data == "pdf":
        data = generate_pdf_transcriptions()
    elif type_of_data == "audio":
        data = generate_audio_transcriptions()
    elif type_of_data == "wiki":
        data = generate_wiki_transcriptions()
    elif type_of_data == "youtube":
        data = generate_youtube_transcriptions()
    else:
        return
    vector_store.add_documents(data)

# Simple fetch data using single query, uses vector.similarity_search to search related documents
def get_data(query):
    vector_store = establish_connection()
    docs = vector_store.similarity_search(query=query, k=1)
    return docs[0].page_content

def get_retriever():
    vector_store = establish_connection(collection_name="wiki_embed")
    retriever = vector_store.as_retriever()
    return retriever

# collection_name = input("Enter collection name: ")
# type_of_data = input("Enter type of data name: ")
# store_embeddings(collection_name, type_of_data)