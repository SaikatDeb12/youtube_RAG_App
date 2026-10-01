from langchain_community.document_loaders import YoutubeLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def getVideoID(url: str) -> str:
    id = YoutubeLoader.extract_video_id(url)
    return id


def getTranscriptChunks(url: str) -> list[Document]:
    loader = YoutubeLoader.from_youtube_url(url)
    docs = loader.load()
    transcript = docs[0].page_content

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.create_documents([transcript])
