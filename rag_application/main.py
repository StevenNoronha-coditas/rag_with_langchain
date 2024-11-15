from dotenv import load_dotenv
from utilities.simple_rag import simple_rag_call
load_dotenv()


print(simple_rag_call("What is the dream of the speaker"))
