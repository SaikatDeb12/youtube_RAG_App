import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone

load_dotenv()

OPEN_AI_KEY = os.getenv("OPEN_AI_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
HF_MODEL = os.getenv("HUGGINGFACE_MODEL")

if not OPEN_AI_KEY:
    raise ValueError("OPEN_AI_KEY is missing!")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is missing!")
if not PINECONE_INDEX_NAME:
    raise ValueError("PINECONE_INDEX_NAME is missing!")
if not HF_MODEL:
    raise ValueError("HF_MODEL is missing!")

pc = Pinecone(PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

llm = HuggingFaceEndpoint(model="google/gemma-4-31B-it")
model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template="""
        You are a helpful assistant.

        Answer only using the provided context

        If the context does not contain enough information to answer the question, then just say: "I don't know."

        context:
        {context}

        question:
        {question}
    """,
    input_variables=["context", "question"],
)

parser = StrOutputParser()

