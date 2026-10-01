from fastapi import APIRouter, HTTPException

from schemas.APISchemas import RequestSchema, ResponseSchema
from services.llm import generateResult
from services.vector_store import isVideoExists, retrieveChunks, storeVideo
from services.youtube import getTranscriptChunks, getVideoID

router = APIRouter()


@router.post("/ask", response_model=ResponseSchema)
def ask(req: RequestSchema):
    try:
        video_id = getVideoID(req.url)
        if not isVideoExists(video_id):
            print("Video not found. Creating embeddings...")
            chunks = getTranscriptChunks(req.url)
            storeVideo(
                chunks,
                video_id,
            )
            print("Video stored in Pinecone.")
        else:
            print("Video already exists. Skipping embeddings.")

        retrieved_docs = retrieveChunks(
            video_id,
            req.query,
        )

        context = "\n\n".join(doc.page_content for doc in retrieved_docs)
        # print(context)
        response = generateResult(context, req.query)

        return ResponseSchema(result=response)
    except Exception as e:
        print("error : ", repr(e))
        raise HTTPException(500, detail=str(e))
