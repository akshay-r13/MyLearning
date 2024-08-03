from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, trim_messages
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from operator import itemgetter

model  = ChatGroq(model="llama3-70b-8192")

prompt_template = ChatPromptTemplate(
    [
        ("system", "You are a useful AI agent called Kora. Help the user by answering questions in {language}"),
        MessagesPlaceholder(variable_name="messages")
    ]
)

trimmer = trim_messages(

    
    max_tokens=65,
    strategy="last",
    token_counter=model,
    include_system=True,
    allow_partial=False,
    start_on="human"
)



chain = (
    RunnablePassthrough.assign(messages=itemgetter("messages") | trimmer)
    | prompt_template
    | model
    | StrOutputParser()
)

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

with_messages_history = RunnableWithMessageHistory(chain, get_session_history, input_messages_key="messages")

configuration = {"configurable": {"session_id": "test-1"}}

messages = [
    SystemMessage(content="you're a good assistant"),
    HumanMessage(content="hi! I'm bob"),
    AIMessage(content="hi!"),
    HumanMessage(content="I like vanilla ice cream"),
    AIMessage(content="nice"),
    HumanMessage(content="whats 2 + 2"),
    AIMessage(content="4"),
    HumanMessage(content="thanks"),
    AIMessage(content="no problem!"),
    HumanMessage(content="having fun?"),
    AIMessage(content="yes!"),
]

response = with_messages_history.invoke(
        {
                "language": "Spanish",
                "messages": messages + [HumanMessage(content="what's my name?")]
        },
        config=configuration
    )

print(response)

config = {"configurable": {"session_id": "abc15"}}

for r in with_messages_history.stream(
    {
        "messages": [HumanMessage(content="hi! I'm todd. tell me a joke")],
        "language": "English",
    },
    config=config,
):
    print(r, end="")