import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from langchain_community.document_loaders import YoutubeLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone
from pydantic import BaseModel

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
HF_MODEL = os.getenv("HUGGINGFACE_MODEL")

if not OPENAI_API_KEY:
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

# llm = HuggingFaceEndpoint(model="google/gemma-4-31B-it")
# model = ChatHuggingFace(llm=llm)
model = ChatOpenAI(model="gpt-6-luna")

parser = StrOutputParser()

app = FastAPI(title="Simple RAG App")


class RequestSchema(BaseModel):
    url: str
    query: str


class ResponseSchema(BaseModel):
    result: str


@app.get("/")
def root():
    return {"message": "Youtube chatApp is running..."}


@app.post("/ask", response_model=ResponseSchema)
def ask(req: RequestSchema):
    try:
        loader = YoutubeLoader.from_youtube_url(req.url)
        docs = loader.load()

        if not docs:
            raise HTTPException(status_code=400, detail="not able to load transcript")

        transcript = docs[0].page_content

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.create_documents(texts=[transcript])

        # creating namespace for pinecone
        # helps to isolate vector data within a single index
        # for now, using the default namespace

        # in embeddings we can also use pinecone's own embeddings
        vectorStore = PineconeVectorStore(index=index, embedding=embeddings)
        vectorStore.add_documents(documents=chunks)

        retrievedDocs = vectorStore.similarity_search(req.query, k=4)
        # Till here we have all the related embeddings

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

        context = " ".join(doc.page_content for doc in retrievedDocs)

        chain = prompt | model | parser
        response = chain.invoke({"context": context, "question": req.query})

        return ResponseSchema(result=response)

    except HTTPException:
        raise
    except Exception as e:
        print("err type: ", type(e))
        print("err", repr(e))
        raise HTTPException(status_code=500, detail=str(e))
