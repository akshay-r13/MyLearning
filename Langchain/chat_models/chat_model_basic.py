from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

model = ChatGroq(model="llama3-70b-8192")

result = model.invoke("What is your name")

print("Result object:")
print(result)
print("Result object content:")
print(result.content)