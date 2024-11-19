from langchain_core.prompts.prompt import PromptTemplate
from langchain_groq.chat_models import ChatGroq
from langchain_core.output_parsers import JsonOutputParser
from typing import List, Optional
import pprint, os
from dotenv import load_dotenv
load_dotenv()

context = """Hollywood's movies showcase the pinnacle of cinematic achievement, blending exceptional storytelling, groundbreaking visuals, and stellar performances. Topping the list, Avatar remains a masterpiece of technological innovation, transporting audiences to Pandora with its immersive 3D visuals and a story centered on environmentalism and interplanetary conflict. Titanic, another James Cameron classic, is a timeless tale of love and tragedy aboard the ill-fated ship, capturing hearts globally with its emotional depth and unforgettable soundtrack. Christopher Nolan's The Dark Knight redefined superhero movies, offering a gritty and complex narrative, highlighted by Heath Ledger's Oscar-winning portrayal of the Joker. Marvel Studios' Avengers: Endgame became a cultural phenomenon, masterfully concluding a decade-long saga with its thrilling battles and heartfelt moments, making it the highest-grossing superhero film ever. Lastly, Inception, another Nolan creation, is a mind-bending journey into dreams within dreams, celebrated for its intricate plot, stunning visuals, and Hans Zimmer's iconic score. These films not only dominated the box office but also left indelible marks on pop culture, pushing the boundaries of what cinema can achieve. From romance and action to fantasy and innovation, Hollywood's finest have set benchmarks that continue to inspire filmmakers and captivate audiences worldwide."""
from pydantic import BaseModel, Field

class Movie(BaseModel):
    movie_name: str = Field(..., description="Insert movie name here")
    movie_director: Optional[str]  = Field(description="Insert movie director name if it exists.")

class MoviesCollection(BaseModel):
    movies: List[Movie] = Field(..., description="Insert a list of movies, that have the movie structure")
    

prompt = """
        You are an helpful assistant, that extracts data from the given context, using the specified format
        Context: {context}
        Format: {format_instructions}
"""

parser = JsonOutputParser(pydantic_object=MoviesCollection)

prompt_template = PromptTemplate(
    template=prompt
)

llm = ChatGroq(
    model_name="llama3-70b-8192",
    groq_api_key=os.environ.get("GROQ_API_KEY")
)

chain = prompt_template | llm | parser

response = chain.invoke({"context": context, "format_instructions": parser.get_format_instructions()})
print(type(response))
pprint.pprint(response)
