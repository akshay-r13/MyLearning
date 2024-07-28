from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.schema.output_parser import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel

load_dotenv()

# Declare Model
model = ChatGroq(model="llama3-70b-8192")

# Starting Prompt
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are an AI assistant which helps users make posts on various social media platforms"),
    ("human", "Hey help me with revised content for my post. My post is: {post}")
])

# Parallel prompts
def create_parallel_chain(messages):
    prompt_template = ChatPromptTemplate.from_messages(messages)
    return prompt_template | model | StrOutputParser()


insta_chain = create_parallel_chain(messages=[
    ("system", "You are an AI assistant which helps users make post to instagram"),
    ("human", "Give me a revised version of my post appropriate for Instagram. Here's my post: {post}.")
])

x_chain = create_parallel_chain(messages=[
    ("system", "You are an AI assistant which helps users make post to twitter"),
    ("human", "Give me a revised version of my post appropriate for Twitter. Here's my post: {post}.")
])

linkedin_chain = create_parallel_chain(messages=[
    ("system", "You are an AI assistant which helps users make post to LinkedIn"),
    ("human", "Revise my post to be appropriate for posting on Linkedin. Here's my post: {post}.")
])

# Post processing function
def process_output(chains_output):
    return f"""
    Insta post content:
    {chains_output["branches"]["instagram"]}

    X post content:
    {chains_output["branches"]["x"]}

    LinkedIn post content:
    {chains_output["branches"]["linkedin"]}
    """

chain = (
    prompt_template 
    | model 
    | StrOutputParser()
    | RunnableParallel(branches={"instagram": insta_chain, "x": x_chain, "linkedin": linkedin_chain})
    | RunnableLambda(lambda x: process_output(x))
)

result = chain.invoke({"post": "I have bought a new car"})

print(result)