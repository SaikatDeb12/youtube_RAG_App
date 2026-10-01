import os

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
HF_TOKEN = os.getenv("HF_TOKEN")
HF_MODEL = os.getenv("HUGGINGFACE_MODEL")
CHAT_MODEL = os.getenv("CHAT_MODEL")

if not OPENAI_API_KEY:
    raise ValueError("OPEN_AI_KEY is missing!")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is missing!")
if not HF_TOKEN:
    raise ValueError("HUGGINGFACE_API_TOKEN is missing!")
if not PINECONE_INDEX_NAME:
    raise ValueError("PINECONE_INDEX_NAME is missing!")
if not HF_MODEL:
    raise ValueError("HF_MODEL is missing!")
if not CHAT_MODEL:
    raise ValueError("CHAT_MODEL is missing!")
