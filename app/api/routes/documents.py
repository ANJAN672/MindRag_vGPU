from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import List
import os
import uuid # Although not used in current upload logic, keep if needed elsewhere
import shutil
# Assuming process_document exists and handles document processing
from app.core.document_processor import process_document
import logging # Import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/documents",
    tags=["documents"],
)

# Directory to store uploaded documents
UPLOAD_DIR = "data/documents"
# Ensure the upload directory exists
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/upload")
async def upload_documents(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    Upload and process documents (PDF, DOCX, PPTX, TXT)
    The processing is offloaded to a background task.
    """
    logger.info(f"Received upload request for file: {file.filename}")

    if not file.filename: # Check if filename is empty
        logger.error("No file provided in upload request.")
        raise HTTPException(status_code=400, detail="No file provided")

    # Check file extension
    file_extension = os.path.splitext(file.filename)[1].lower()
    supported_extensions = [".pdf", ".docx", ".pptx", ".txt"]
    if file_extension not in supported_extensions:
        logger.warning(f"Unsupported file format received: {file.filename} (Extension: {file_extension})")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file format. Supported formats: {', '.join(supported_extensions)}"
        )

    # Use the original filename directly for storage
    # This means if a file with the same name is uploaded, it will overwrite the old one.
    stored_filename = file.filename
    file_path = os.path.join(UPLOAD_DIR, stored_filename)

    # If file already exists, remove it first to ensure a clean overwrite
    if os.path.exists(file_path):
        logger.info(f"Existing file found: {file_path}. Attempting to remove.")
        try:
            os.remove(file_path)
            logger.info(f"Successfully removed existing file: {file_path}")
        except OSError as e: # Catch specific OS error for file operations
             logger.error(f"Error removing existing file {file_path}: {e}", exc_info=True)
             # Decide if this should be a blocking error or just a warning
             # For now, let's allow the upload to proceed, which will likely overwrite anyway
             pass # Allow to proceed, shutil.copyfileobj might handle overwrite

    # Save the new file to the upload directory
    try:
        with open(file_path, "wb") as buffer:
            # Use file.file.read() in chunks for potentially large files
            # shutil.copyfileobj is good, but reading in chunks is safer for very large files
            # For simplicity, keeping shutil.copyfileobj as it was, but be aware of large files.
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"Successfully saved uploaded file to: {file_path}")
    except Exception as e:
        logger.error(f"Error saving uploaded file {file.filename}: {e}", exc_info=True)
        # Clean up partially written file if error occurred during save
        if os.path.exists(file_path):
             try:
                 os.remove(file_path)
                 logger.info(f"Cleaned up partial file after save error: {file_path}")
             except Exception as cleanup_e:
                 logger.error(f"Error during cleanup of partial file {file_path}: {cleanup_e}", exc_info=True)

        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")


    # Process document in background
    # This function (process_document) is where the main RAG processing happens.
    # If this function crashes or takes too long without proper error handling,
    # it could lead to ECONNRESET errors on the client side if the background
    # task doesn't complete cleanly or the main process exits unexpectedly.
    logger.info(f"Adding background task to process document: {file_path}")
    background_tasks.add_task(process_document, file_path, file.filename)

    # Return a 202 Accepted response as processing is happening in the background
    uploaded_file_info = {
        "original_name": file.filename,
        "stored_name": stored_filename, # In this case, same as original_name
        "status": "processing", # Indicate that processing has started
        # You might want to add a unique ID here if process_document generates one
        # For now, relying on filename as the identifier in the frontend
        "id": stored_filename # Using filename as a simple ID for the frontend list
    }

    return JSONResponse(
        status_code=202, # 202 Accepted is appropriate for background processing
        content={
            "message": f"Uploaded document: {file.filename}. Processing started.",
            "file": uploaded_file_info # Return info about the uploaded file
        }
    )

@router.get("/")
async def list_documents():
    """
    List all documents currently stored in the upload directory.
    Assumes files in this directory are processed or processing.
    Returns filename and size.
    """
    logger.info("Received request to list documents.")
    documents = []

    # Iterate through files in the upload directory
    for filename in os.listdir(UPLOAD_DIR):
        file_path = os.path.join(UPLOAD_DIR, filename)
        # Ensure it's a file, not a directory
        if os.path.isfile(file_path):
            try:
                # Get file size
                file_size = os.path.getsize(file_path)
                documents.append({
                    # Using filename as a simple ID for the frontend list
                    "id": filename,
                    "filename": filename,
                    "size": file_size,
                    # You might want to add a 'status' here if you track it separately
                    # For now, assuming presence in UPLOAD_DIR means 'processed' or 'processing'
                })
            except OSError as e:
                 logger.error(f"Error getting info for file {filename}: {e}", exc_info=True)
                 # Optionally, skip files that cause errors or add them with an error status
                 documents.append({
                    "id": filename,
                    "filename": filename,
                    "size": "Error",
                    "status": "error",
                    "message": "Could not retrieve file info"
                 })


    logger.info(f"Returning list of {len(documents)} documents.")
    return {"documents": documents}

@router.delete("/{filename}")
async def delete_document(filename: str):
    """
    Delete a document by filename.
    Also needs to handle removal of associated embeddings.
    """
    logger.info(f"Received request to delete document: {filename}")
    file_path = os.path.join(UPLOAD_DIR, filename)

    # Check if the file exists before attempting deletion
    if not os.path.exists(file_path):
        logger.warning(f"Attempted to delete non-existent document: {filename}")
        raise HTTPException(status_code=404, detail="Document not found")

    try:
        # Attempt to remove the file
        os.remove(file_path)
        logger.info(f"Successfully deleted file: {file_path}")

        # TODO: Implement logic here to also remove associated embeddings
        # This is crucial for maintaining the integrity of your RAG system.
        # Example (pseudocode):
        # try:
        #     rag_engine = get_rag_engine() # Assuming you can get the engine instance
        #     rag_engine.remove_embeddings_for_document(filename)
        #     logger.info(f"Removed embeddings for document: {filename}")
        # except Exception as e:
        #     logger.error(f"Error removing embeddings for {filename}: {e}", exc_info=True)
        #     # Decide how to handle this error - maybe return a partial success or full error
            # For now, just logging and continuing to return success for file deletion

        return {"message": f"Document {filename} deleted successfully"}

    except OSError as e: # Catch specific OS error for file operations
        logger.error(f"Error deleting file {file_path}: {e}", exc_info=True)
        # Raising HTTPException for file deletion errors
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")
    except Exception as e: # Catch any other unexpected errors during deletion process
        logger.error(f"An unexpected error occurred during deletion of {filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

