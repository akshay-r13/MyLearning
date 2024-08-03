from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.schema.output_parser import StrOutputParser
from langchain_core.runnables import RunnableLambda

load_dotenv()

# Declare Model
model = ChatGroq(model="llama3-70b-8192")

# Declare custom runnables
upper_case = RunnableLambda(lambda x: x.upper())
word_count = RunnableLambda(lambda x: f"Word count: {len(x.split(" "))}\n{x}")

# Declare prompt Template
prompt_template = ChatPromptTemplate.from_template("Give me info about {topic} in relation to {country}")

chain = prompt_template | model | StrOutputParser() | upper_case | word_count

result = chain.invoke({"topic": "Sports", "country": "India"})

print(result)