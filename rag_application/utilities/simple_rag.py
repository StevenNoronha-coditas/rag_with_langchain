from langchain import hub
from utilities.db_operations import get_retriever
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough

def simple_rag_call(query):
    prompt = hub.pull("rlm/rag-prompt")
    retriever = get_retriever()

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    llm = ChatGroq(model="llama3-8b-8192")

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    result = rag_chain.invoke(query)
    return result