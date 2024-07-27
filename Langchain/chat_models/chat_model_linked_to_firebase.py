import os
from dotenv import load_dotenv
from google.cloud import firestore
from langchain_google_firestore import FirestoreChatMessageHistory
from langchain_groq import ChatGroq

load_dotenv()

SESSION_ID = "user_session_1"
FIREBASE_CHAT_HISTORY_COLLECTION_NAME = "chat_history"

print("Initializing Firestore Client")
client = firestore.Client(project=os.environ["FIREBASE_PROJECT_ID"])

print("Initializing Firestore Chat Message History")
chat_history = FirestoreChatMessageHistory(
    session_id=SESSION_ID,
    collection=FIREBASE_CHAT_HISTORY_COLLECTION_NAME,
    client=client
)
print("Chat History initialized.")
print("Current chat history: ", chat_history.messages)

model = ChatGroq(model="llama3-70b-8192")

while True:
    user_message = input("You: ")
    if not user_message or user_message == "exit":
        break
    chat_history.add_user_message(message=user_message)
    result = model.invoke(chat_history.messages)
    ai_message = result.content
    chat_history.add_ai_message(message=ai_message)
    print("Model: ", ai_message)


print("Chat history object:")
print(chat_history)