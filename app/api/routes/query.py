from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional, Dict, Any # Import Dict and Any for return type
# Assuming RAGEngine and get_rag_engine exist and are correctly implemented
from app.core.rag_engine import RAGEngine, get_rag_engine
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


router = APIRouter(
    prefix="/query",
    tags=["query"],
)

# Pydantic model for the query request body
class QueryRequest(BaseModel):
    query: str
    # Optional list of document filenames to search in.
    # If None or empty, the RAG engine should search all available documents.
    documents: Optional[List[str]] = None
    # Parameters for the language model
    max_tokens: Optional[int] = 1024
    temperature: Optional[float] = 0.7
    detailed: Optional[bool] = False
    fast_mode: Optional[bool] = False # Add fast_mode parameter


# Pydantic model for the query response body (re-added)
class QueryResponse(BaseModel):
    answer: str
    sources: List[dict] # List of source documents/chunks used for the answer
    processing_time: float # Time taken to process the query in seconds
    is_complex: Optional[bool] = False # Optional flag if needed by the frontend


@router.post("/", response_model=QueryResponse) # Change return type hint back to QueryResponse
async def query_documents(
    request: QueryRequest,
    # Dependency injection: Get an instance of RAGEngine
    rag_engine: RAGEngine = Depends(get_rag_engine)
):
    """
    Query the documents using the RAG system.
    Takes a query string and optional parameters, returns an answer and sources.
    The RAG engine will print streaming output to the backend console.
    """
    logger.info(f"Received non-streaming query request: '{request.query}' with documents: {request.documents}")

    # Basic validation for the query string
    if not request.query or not request.query.strip():
        logger.warning("Received empty or whitespace-only query.")
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    start_time = time.time() # Record start time for processing time calculation

    try:
        # Prepare the document filter: pass None if the list is empty or None
        document_filter = request.documents if request.documents and len(request.documents) > 0 else None
        logger.info(f"Using document filter: {document_filter}")

        # Call the RAG engine's query method which returns a generator
        result_generator = rag_engine.query(
            query=request.query,
            document_filter=document_filter,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            detailed=request.detailed,
            fast_mode=request.fast_mode # Pass fast_mode
        )

        # Process the generator to collect all chunks
        answer_chunks = []
        sources = []
        processing_time = 0
        is_complex = False
        
        # Consume the generator to get all chunks
        for chunk in result_generator:
            chunk_type = chunk.get("type", "")
            
            if chunk_type == "text":
                answer_chunks.append(chunk.get("content", ""))
            elif chunk_type == "metadata":
                metadata = chunk.get("data", {})
                sources = metadata.get("sources", [])
                processing_time = metadata.get("processing_time", 0)
                is_complex = metadata.get("is_complex", False)
            elif chunk_type == "error":
                # If there's an error, raise an exception
                error_message = chunk.get("message", "Unknown error in RAG processing")
                logger.error(f"Error from RAG engine: {error_message}")
                raise HTTPException(status_code=500, detail=error_message)

        # Combine all answer chunks into a single answer
        answer = "".join(answer_chunks) if answer_chunks else "No answer found."
        
        end_time = time.time() # Record end time
        if processing_time == 0:
            processing_time = end_time - start_time # Calculate processing time if not provided by RAG engine
        
        logger.info(f"Query processed successfully in {processing_time:.4f} seconds.")

        # Prepare the response content
        response_content = {
            "answer": answer,
            "sources": sources,
            "processing_time": processing_time,
            "is_complex": is_complex
        }


        return QueryResponse(**response_content) # Return using the Pydantic model

    except Exception as e:
        # Catch any exceptions during the query process and return a 500 error
        logger.error(f"Error processing query '{request.query}': {e}", exc_info=True)
        # Re-raise as HTTPException with a 500 status code
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

# Add a simple test endpoint for verification
@router.get("/test")
async def test_query():
    """
    Test endpoint to verify the query route is working.
    Does not perform a full RAG query.
    """
    logger.info("Received request for test query endpoint.")
    return {"status": "Query endpoint is working"}
