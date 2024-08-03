from dotenv import load_dotenv
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_groq import ChatGroq

load_dotenv()

model = ChatGroq(model="llama3-70b-8192")

messages = [
    SystemMessage(content="Solve the following math problems"),
    HumanMessage(content="What is 9 divided by 3")
]

result = model.invoke(messages)

print("Result object:")
print(result)
print("-----")
print("Result object content:")
print(result.content)