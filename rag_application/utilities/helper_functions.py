from dotenv import load_dotenv
from groq import Groq
import os
from langchain.schema import Document  
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.document_loaders import AssemblyAIAudioTranscriptLoader, PyPDFLoader, WikipediaLoader, YoutubeLoader

load_dotenv()

def generate_transcriptions_using_groq():
    client = Groq()
    filename = os.path.dirname(__file__) + "/audio.mp3" 
    docs = []
    with open(filename, "rb") as file:
        transcription = client.audio.transcriptions.create(
        file=(filename, file.read()), 
        model="whisper-large-v3-turbo", 
        language="en",  
        temperature=0.0  
        )
        docs = [Document(page_content=transcription.text)]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    return splits

def generate_audio_transcriptions():
    audio_file = os.path.dirname(__file__) + "\data_sources\audio.mp3" 
    loader = AssemblyAIAudioTranscriptLoader(file_path=audio_file)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    return splits

def generate_pdf_transcriptions():
    pdf_file = os.path.dirname(__file__) + "\data_sources\knowledge.pdf"
    loader = PyPDFLoader(file_path=pdf_file)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)
    return splits

def generate_wiki_transcriptions():
    wiki_search = "Mercedes Maybach"
    loader = WikipediaLoader(query=wiki_search, load_max_docs=4)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)
    return splits

def generate_youtube_transcriptions():
    utube_url = "https://www.youtube.com/watch?v=Y_O-x-itHaU"
    loader = YoutubeLoader.from_youtube_url(utube_url, add_video_info=False)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)
    return splits

def get_embedding_model():
    model_name = "BAAI/bge-small-en"
    hf = HuggingFaceBgeEmbeddings(model_name=model_name)
    return hf