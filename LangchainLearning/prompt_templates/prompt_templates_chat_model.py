from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

load_dotenv()

model = ChatGroq(model="llama3-70b-8192")


# Part 1: Generating prompt from prompt template object
template = "Give me info about {topic} in relation to {country}"
prompt_template = ChatPromptTemplate.from_template(template)

prompt = prompt_template.invoke({"topic": "Sports", "country": "India"})

result = model.invoke(prompt)

print(result.content)

# Part 2: Templating across different messages in a chat history
messages = [
    ("system", "You are a helpful AI assistant who helps user on the topic of {topic}"),
    ("human", "Hi I am from {country}. How can you help me")
]

prompt_template = ChatPromptTemplate.from_messages(messages)
prompt = prompt_template.invoke({"topic": "Technology", "country": "UK"})
result = model.invoke(prompt)
print(result.content)