from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from fastapi import FastAPI
from langserve import add_routes

# Load environment variables
load_dotenv()

# Load model
model = ChatGroq(model="llama3-70b-8192")
# Initiate parser
string_parser = StrOutputParser()

# Creating a prompt template to translate text
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "Translate the following text into {language}"),
        ("user", "{text}")
    ]
)

# Chain definition
chain = prompt_template | model | string_parser

# App definition
app = FastAPI(
    title="Simple Translation app",
    version="1.0",
    description="A simple app which takes user text and translate to a specific language"
)

add_routes(
    app,
    chain,
    path="/translate"
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)