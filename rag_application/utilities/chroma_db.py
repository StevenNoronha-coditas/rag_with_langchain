from langchain_chroma import Chroma
from rag_application.utilities.helper_functions import generate_transcriptions, get_embedding_model
from langchain import hub
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

# Used as trial to store embeddings using chroma_db
def rag_using_chromdb():
    hf = get_embedding_model()
    persistent_directory = "./chroma_storage"

    splits = generate_transcriptions()

    vectorstore = Chroma.from_documents(
        documents=splits,  
        embedding=hf,
        persist_directory=persistent_directory
    )

    retriever = vectorstore.as_retriever()
    prompt = hub.pull("rlm/rag-prompt")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    llm = ChatGroq(model="llama3-8b-8192")

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print(rag_chain.invoke("What is the dream the speaker is talking about"))