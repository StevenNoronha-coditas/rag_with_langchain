import streamlit as st
import requests
import json
from typing import List, Dict

def initialize_session_state():
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'session_id' not in st.session_state:
        st.session_state.session_id = None

def display_chat_messages():
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]
        with st.chat_message(role):
            st.write(content)

def send_message(user_input: str) -> Dict:
    url = "http://localhost:5000/chat"
    
    payload = {
        "query": user_input,
        "session_id": st.session_state.session_id
    }
    
    response = requests.post(url, json=payload)
    return response.json()

def main():
    st.title("RAG Chatbot")
    initialize_session_state()

    # Chat interface
    if st.session_state.messages:
        display_chat_messages()

    # User input
    user_input = st.chat_input("Type your message here...")
    
    if user_input:
        # Display user message
        with st.chat_message("human"):
            st.write(user_input)
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "human", "content": user_input})
        
        try:
            # Get response from backend
            response_data = send_message(user_input)
            
            # Update session ID if it's a new conversation
            if st.session_state.session_id is None:
                st.session_state.session_id = response_data['session_id']
            print(st.session_state.session_id)
            
            # Display AI response
            with st.chat_message("ai"):
                st.write(response_data['response'])
            
            # Add AI response to chat history
            st.session_state.messages.append({"role": "ai", "content": response_data['response']})
            
            # Update full chat history from backend
            st.session_state.messages = [
                {"role": msg["role"], "content": msg["content"]} 
                for msg in response_data['chat_history']
            ]
            
        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 