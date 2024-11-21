from dotenv import load_dotenv
from utilities.simple_rag import simple_rag_call
load_dotenv()

# Initialize with no session ID for first call
response, chat_history, session_id = simple_rag_call("Tell me about mercedes")
print("Response:", response)
print("Session ID:", session_id)

# Use the same session ID for subsequent calls to maintain conversation context
response2, chat_history2, _ = simple_rag_call("Can you elaborate on that?", session_id)
print("\nFollow-up Response:", response2)

print("\nChat History:")
for message in chat_history2:
    print(f"{message.type}: {message.content}")
