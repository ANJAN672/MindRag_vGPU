from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
# Assuming these modules exist and contain FastAPI routers
from app.api.routes import documents, query
import os # Import os module
import glob # Import glob module

# Initialize the FastAPI application
app = FastAPI(
    title="MindRAG",
    description="A powerful Retrieval-Augmented Generation (RAG) system",
    version="0.1.0",
)

# Configure CORS (Cross-Origin Resource Sharing) middleware
# This allows your frontend running on a different port (like 3000)
# to make requests to your backend (on port 8000).
# WARNING: allow_origins=["*"] is permissive and should be restricted
# to your frontend's specific origin(s) in a production environment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (use specific origins in production)
    allow_credentials=True, # Allows cookies and authorization headers
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Include routers for different parts of the API
# The 'prefix="/api"' means all routes defined in documents.router
# and query.router will be prefixed with /api.
# E.g., a route /upload in documents.router becomes /api/upload.
app.include_router(documents.router, prefix="/api")
app.include_router(query.router, prefix="/api")

# Root endpoint
@app.get("/")
async def root():
    """
    Root endpoint providing a welcome message and link to docs.
    """
    return {
        "message": "Welcome to MindRAG API",
        "docs": "/docs", # Link to the auto-generated OpenAPI docs
    }

# Health check endpoint
# NOTE: Removed duplicate definitions of health_check.
# Only one function definition per endpoint path is needed.
@app.get("/api/health")
async def health_check():
    """
    Health check endpoint for the frontend to verify backend status.
    Also checks if the documents directory exists and contains files.
    """
    # Check if documents directory exists and has files
    documents_dir = "data/documents"
    has_documents = False

    # Use glob to find any files (not directories) within the documents_dir
    # This is a more reliable check than just os.listdir
    if os.path.exists(documents_dir):
        # Check if there's at least one file matching any pattern (e.g., *)
        # or specifically look for files if you have known extensions
        # For a simple check, glob(os.path.join(documents_dir, '*')) is sufficient
        files = glob.glob(os.path.join(documents_dir, '*'))
        # Filter out directories if necessary, though glob('*') usually gets files/dirs
        # A more precise check might be:
        # files = [f for f in glob.glob(os.path.join(documents_dir, '*')) if os.path.isfile(f)]
        has_documents = len(files) > 0

    return {
        "status": "healthy",
        "has_documents": has_documents
    }

# Uvicorn server runner
# This block allows you to run the application using 'python app/main.py'
# It's typically used for development. In production, you'd use a separate
# uvicorn command or a process manager.
if __name__ == "__main__":
    import uvicorn
    # Running with reload=True is good for development as it restarts the server
    # on code changes.
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
