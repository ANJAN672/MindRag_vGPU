# MindRAG - Open-Source RAG System

MindRAG is a powerful Retrieval-Augmented Generation (RAG) system that allows you to upload documents, process them with state-of-the-art AI models, and get accurate answers to your questions based on the content of those documents.

## Features

- **Document Processing**: Upload and process PDF, Word, PowerPoint, and text documents
- **Semantic Search**: Find relevant information using advanced embedding models
- **AI-Powered Answers**: Get comprehensive answers based on your documents
- **Local Execution**: All processing happens on your machine for privacy
- **Open Source**: Built with open-source models and libraries

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/mindrag.git
   cd mindrag
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

### Running the Application

1. Start the backend server:
   ```
   python run.py
   ```

2. The API will be available at `http://localhost:8000`
3. API documentation is available at `http://localhost:8000/docs`

## Usage

### Uploading Documents

Upload documents using the `/documents/upload` endpoint. Supported formats:
- PDF (.pdf)
- Microsoft Word (.docx)
- Microsoft PowerPoint (.pptx)
- Text files (.txt)

### Querying Documents

Once documents are uploaded and processed, you can query them using the `/query` endpoint.

## API Endpoints

- `POST /documents/upload`: Upload documents
- `GET /documents`: List all uploaded documents
- `DELETE /documents/{filename}`: Delete a document
- `POST /query`: Query the documents

## Technical Details

MindRAG uses:
- FastAPI for the backend API
- Hugging Face's Zephyr model for text generation
- LangChain for the RAG pipeline
- FAISS for vector storage and similarity search
- Sentence Transformers for document embeddings

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Hugging Face for providing open-source models
- LangChain for the RAG framework
- FAISS for efficient similarity search