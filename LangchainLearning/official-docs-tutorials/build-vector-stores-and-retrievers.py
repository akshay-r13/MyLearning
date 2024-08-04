from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List
from langchain_core.runnables import RunnableLambda

model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
hf = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

documents = [
    Document(
        page_content="Dogs are great companions, known for their loyalty and friendliness.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Cats are independent pets that often enjoy their own space.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Goldfish are popular pets for beginners, requiring relatively simple care.",
        metadata={"source": "fish-pets-doc"},
    ),
    Document(
        page_content="Parrots are intelligent birds capable of mimicking human speech.",
        metadata={"source": "bird-pets-doc"},
    ),
    Document(
        page_content="Rabbits are social animals that need plenty of space to hop around.",
        metadata={"source": "mammal-pets-doc"},
    ),
]

vectorstore = Chroma.from_documents(
    documents,
    embedding=HuggingFaceEmbeddings(),
)

print(vectorstore.similarity_search("cat"))

print(vectorstore.similarity_search_with_score("cat"))

embedding = HuggingFaceEmbeddings().embed_query("cat")
print(vectorstore.similarity_search_by_vector(embedding))

retriever = RunnableLambda(vectorstore.similarity_search).bind(k=1)  # select top result

print(retriever.batch(["cat", "shark"]))

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1},
)

print(retriever.batch(["cat", "shark"]))