from langchain import hub
from utilities.db_operations import get_retriever
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from utilities.chat_history import initialize_chat_history, add_message_to_history, get_recent_chat_history

def simple_rag_call(query, session_id=None):
    chat_history = initialize_chat_history(session_id)
    
    add_message_to_history(chat_history, "human", query)
    
    recent_messages = get_recent_chat_history(chat_history)
    
    retriever = get_retriever()

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant. Use the following context to answer the question, "
                  "taking into account the chat history if relevant: {context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])
    
    llm = ChatGroq(model="llama3-8b-8192")

    rag_chain = (
        {
            "context": retriever | format_docs,
            "chat_history": lambda _: recent_messages,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    result = rag_chain.invoke(query)
    
    add_message_to_history(chat_history, "ai", result)
    
    return result, chat_history.messages, chat_history._session_id