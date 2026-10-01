from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

from config import PINECONE_API_KEY, PINECONE_INDEX_NAME

pc = Pinecone(api_key=PINECONE_API_KEY)
indexName = PINECONE_INDEX_NAME or "sample-index"
index = pc.Index(indexName)

embeddings = OpenAIEmbeddings()

# creating namespace for pinecone
# helps to isolate vector data within a single index
# for now, using the default namespace

# in embeddings we can also use pinecone's own embeddings
vectorStore = PineconeVectorStore(index=index, embedding=embeddings)


# need to extract videoID and check if it already exists
def isVideoExists(videoID: str) -> bool:
    result = index.query(
        vector=[0.0] * 1536,
        top_k=1,
        filter={"video_id": {"$eq": videoID}},
        include_metadata=True,
        include_values=False,
    )
    if len(result.get("matches", [])) > 0:
        return True
    return False


def storeVideo(chunks: list[Document], videoID: str):
    for chunk in chunks:
        chunk.metadata["video_id"] = videoID
    print("Storing video:", videoID)
    print("Metadata:", chunks[0].metadata)
    vectorStore.add_documents(documents=chunks)


def retrieveChunks(videoID: str, query: str):
    return vectorStore.similarity_search(
        query,
        filter={"video_id": {"$eq": videoID}},
    )
