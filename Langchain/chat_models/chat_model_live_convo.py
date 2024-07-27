from dotenv import load_dotenv
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_groq import ChatGroq

load_dotenv()

model = ChatGroq(model="llama3-70b-8192")

chat_history = [
    SystemMessage(content="You are a grammar enhancing AI agent")
]

while True:
    user_message = input("You: ")
    if not user_message or user_message == "exit":
        break
    chat_history.append(HumanMessage(content=user_message))
    result = model.invoke(chat_history)
    ai_message = result.content
    chat_history.append(AIMessage(content=ai_message))
    print("Model: ", ai_message)


print("Chat history object:")
print(chat_history)