from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.schema.output_parser import StrOutputParser

load_dotenv()

# Declare Model
model = ChatGroq(model="llama3-70b-8192")

# Declare prompt Template
prompt_template = ChatPromptTemplate.from_template("Give me info about {topic} in relation to {country}")

chain = prompt_template | model | StrOutputParser()

result = chain.invoke({"topic": "Sports", "country": "India"})

print(result)