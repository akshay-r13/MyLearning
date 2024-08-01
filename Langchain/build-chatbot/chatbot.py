from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

model  = ChatGroq(model="llama3-70b-8192")

prompt_template = ChatPromptTemplate(
    [
        ("system", "You are a useful AI agent called Kora. Help the user by answering questions in {language}"),
        MessagesPlaceholder(variable_name="messages")
    ]
)

chain = prompt_template | model | StrOutputParser()

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

with_messages_history = RunnableWithMessageHistory(chain, get_session_history, input_messages_key="messages")

configuration = {"configurable": {"session_id": "test-1"}}

response = with_messages_history.invoke(
        {
                "language": "Spanish",
                "messages": [HumanMessage("My name is Bob"), AIMessage("Nice to meet you"), HumanMessage("What's your name")]
        },
        config=configuration
    )

print(response)