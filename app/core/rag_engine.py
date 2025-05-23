import os
import time
import logging
from typing import List, Dict, Optional, Any, Generator, Union
import torch
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import login
import os
from dotenv import load_dotenv
import json # Import json to serialize metadata chunk

# Import GPU utilities
from app.core.gpu_utils import device_config

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
# Explicitly set device to CUDA if available, otherwise fallback to CPU
if torch.cuda.is_available():
    DEVICE = "cuda"
    EMBEDDINGS_DEVICE = "cuda"
    logger.info("CUDA available. Using GPU for both LLM and Embeddings.")
else:
    DEVICE = "cpu"
    EMBEDDINGS_DEVICE = "cpu"
    logger.warning("CUDA not available. Using CPU for both LLM and Embeddings.")

# Get GPU layers from environment or default based on device config if CUDA is available
GPU_LAYERS = 0
if DEVICE == "cuda":
    # Use GPU_LAYERS from environment variable, fallback to device_config if not set
    # Default to 32 layers for better performance if not specified
    GPU_LAYERS = int(os.getenv("GPU_LAYERS", device_config.get('gpu_layers', 32)))
    if GPU_LAYERS == 0:
         logger.warning("DEVICE set to cuda but GPU_LAYERS is 0 or not set in environment. LLM will run on CPU layers.")
    else:
         logger.info(f"Using GPU acceleration with {GPU_LAYERS} layers.")

CPU_THREADS = int(os.getenv("CPU_THREADS", "4")) # Default CPU threads

# Directories for storing embeddings and models
EMBEDDINGS_DIR = "data/embeddings"
MODELS_DIR = "data/models"

# Ensure necessary directories exist
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# --- Embeddings Model Initialization ---
logger.info(f"Initializing embeddings model on {EMBEDDINGS_DEVICE} device")

# Use larger batch size for GPU to improve performance
batch_size = 64 if EMBEDDINGS_DEVICE == "cuda" else 32

try:
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": EMBEDDINGS_DEVICE},
        encode_kwargs={
            "normalize_embeddings": True,
            "device": EMBEDDINGS_DEVICE,
            "batch_size": batch_size  # Larger batch size for GPU processing
        },
        show_progress=False  # Disable progress bar for faster processing
    )
    logger.info("Embeddings model initialized successfully.")
except Exception as e:
    logger.error(f"Error initializing embeddings model: {e}", exc_info=True)
    # Handle the error - perhaps raise it or use a mock embeddings model
    raise RuntimeError("Failed to initialize embeddings model.") from e


# --- Language Model (LLM) Configuration ---
# Get configuration from environment variables
HF_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN", "")
MODEL_TYPE = os.getenv("MODEL_TYPE", "local").lower() # Default to local, make lowercase
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "TheBloke/zephyr-7B-beta-GGUF") # Default local model
MODEL_BASENAME = os.getenv("MODEL_BASENAME", "zephyr-7b-beta.Q2_K.gguf") # Default local model file


logger.info(f"LLM Configuration: Model Type: {MODEL_TYPE}, Default Model: {DEFAULT_MODEL}")
logger.info(f"LLM Device: {DEVICE}, GPU Layers: {GPU_LAYERS}, CPU Threads: {CPU_THREADS}")


# Login to Hugging Face if token is provided and not using a local model that doesn't need it
if HF_TOKEN and MODEL_TYPE != "local":
    logger.info("Logging in to Hugging Face Hub")
    try:
        login(token=HF_TOKEN)
        logger.info("Hugging Face Hub login successful.")
    except Exception as e:
        logger.error(f"Hugging Face Hub login failed: {e}", exc_info=True)
        # Decide if login failure should be a fatal error or allow fallback
        # For now, just log and continue, relying on fallback mechanisms
else:
    logger.info("Skipping Hugging Face Hub login (no token or using local model)")


class RAGEngine:
    """
    Core RAG engine class for loading models, vector stores, and processing queries.
    """
    def __init__(self):
        self.embeddings = embeddings # Use the globally initialized embeddings model
        self.llm = self._initialize_llm() # Initialize the language model
        self.prompt_template = self._create_prompt_template() # Create the main prompt template
        self.simple_prompt_template = self._create_simple_prompt_template() # Create a simple prompt template

        # Create a modified prompt template for the chain that doesn't require 'instruction'
        chain_template = """You are an advanced AI assistant similar to ChatGPT that provides comprehensive, natural-sounding answers based on document content. Your goal is to answer questions thoroughly while maintaining a conversational tone.

Document content:
{context}

Question: {query}

RESPONSE GUIDELINES:
1. Provide a comprehensive, detailed answer based on the document content
2. Write in a natural, conversational style like ChatGPT
3. Organize information logically and present it clearly
4. If the document doesn't contain enough information, acknowledge this politely
5. Stay factual and accurate to the document content
6. Don't mention "chunks," "context," or other technical details in your answer
7. Don't use phrases like "according to the document" or "based on the provided information"
8. Integrate information from different parts of the document seamlessly
9. For author information, only mention what is clearly stated in the document

Format your answer using Markdown for better readability:
- Use **bold** for important terms or concepts
- Use bullet points or numbered lists when appropriate
- Use headings (## or ###) for sections if needed
- Include clear paragraph breaks for readability

Answer:"""

        chain_prompt = PromptTemplate(
            input_variables=["context", "query"],
            template=chain_template
        )

        # Initialize LangChain LLMChains with appropriate templates
        # NOTE: LLMChain is deprecated. Consider migrating to LCEL (LangChain Expression Language)
        # e.g., chain = prompt | llm
        # For streaming with LLMChain, you'd typically use .stream() on the chain
        self.llm_chain = LLMChain(prompt=chain_prompt, llm=self.llm)
        self.simple_llm_chain = LLMChain(prompt=self.simple_prompt_template, llm=self.llm)


    def _initialize_llm(self):
        """
        Initialize the LLM based on MODEL_TYPE configuration.
        Supports local models (LlamaCpp) and API-based models (HuggingFaceEndpoint).
        Includes fallback mechanisms.
        """
        llm = None # Initialize llm to None

        try:
            # --- Local Model using llama.cpp ---
            if MODEL_TYPE == "local":
                try:
                    from langchain_community.llms import LlamaCpp

                    model_id = DEFAULT_MODEL
                    model_basename = MODEL_BASENAME
                    model_path = os.path.join(MODELS_DIR, model_basename)

                    # Download the model if the file doesn't exist locally
                    if not os.path.exists(model_path):
                        logger.info(f"Downloading local model {model_id}/{model_basename} to {MODELS_DIR}...")
                        from huggingface_hub import hf_hub_download
                        # hf_hub_download returns the path to the downloaded file
                        model_path = hf_hub_download(
                            repo_id=model_id,
                            filename=model_basename,
                            cache_dir=MODELS_DIR, # Use MODELS_DIR as the cache directory
                            resume_download=True # Resume download if interrupted
                        )
                        logger.info(f"Model downloaded to {model_path}")
                    else:
                        logger.info(f"Local model found at {model_path}, skipping download.")

                    # Initialize the model using LangChain's LlamaCpp wrapper with optimized settings
                    logger.info(f"Initializing LlamaCpp model from {model_path}")

                    # Determine optimal thread count based on CPU cores if not explicitly set
                    num_threads = CPU_THREADS
                    if num_threads <= 0: # If CPU_THREADS is not set or invalid
                         import multiprocessing
                         # Estimate physical cores (usually total_cores / 2)
                         physical_cores = multiprocessing.cpu_count() // 2
                         # Use between 2 and 4 threads, capped by physical cores
                         num_threads = min(4, max(2, physical_cores))
                         logger.info(f"CPU_THREADS not set or invalid, using estimated optimal threads: {num_threads}")
                    else:
                         logger.info(f"Using CPU_THREADS from environment: {num_threads}")


                    # Configure LlamaCpp with GPU acceleration if available and GPU_LAYERS > 0
                    gpu_layers = GPU_LAYERS if DEVICE == "cuda" else 0

                    if gpu_layers > 0:
                        logger.info(f"Using GPU acceleration with {gpu_layers} layers for LlamaCpp")
                    else:
                        logger.info("Using CPU-only mode for LlamaCpp")

                    # Get VRAM size to potentially optimize settings further
                    vram_size_gb = 0
                    if torch.cuda.is_available():
                        try:
                            vram_size_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                            logger.info(f"Detected GPU with {vram_size_gb:.2f} GB VRAM")
                        except Exception as vram_e:
                             logger.warning(f"Could not get GPU VRAM size: {vram_e}")


                    # Optimize context size (n_ctx) and batch size (n_batch) based on device/VRAM
                    # These values significantly impact performance and memory usage.
                    # Adjust these based on your specific GPU and model size.
                    context_size = 4096 # Default context window size
                    batch_size = 512 # Default batch size

                    if DEVICE == "cuda" and vram_size_gb > 0:
                        # Aggressive optimization for 4GB cards like GTX 1650
                        if vram_size_gb < 5:
                            # For GTX 1650, optimize for speed with these settings
                            context_size = 2048 # Keep context smaller
                            batch_size = 2048 # Much larger batch for throughput
                            logger.info(f"Optimizing for GTX 1650 with <5GB VRAM: context_size={context_size}, batch_size={batch_size}")
                        # Optimization for 8GB cards
                        elif vram_size_gb < 9:
                            context_size = 4096 # Standard context
                            batch_size = 2048 # Increased batch
                            logger.info(f"Optimizing for 8GB VRAM: context_size={context_size}, batch_size={batch_size}")
                        # Optimization for 12GB+ cards
                        else:
                            context_size = 8192 # Larger context
                            batch_size = 2048 # Larger batch
                            logger.info(f"Optimizing for 12GB+ VRAM: context_size={context_size}, batch_size={batch_size}")
                    elif DEVICE == "cpu":
                         # Adjust batch size for CPU if needed, typically smaller than GPU
                         batch_size = min(batch_size, 512) # Keep batch size reasonable for CPU
                         logger.info(f"Using CPU settings: context_size={context_size}, batch_size={batch_size}, threads={num_threads}")


                    # Optimize parameters for faster inference
                    llm = LlamaCpp(
                        model_path=model_path,
                        temperature=float(os.getenv("LLM_TEMPERATURE", "0.5")), # Use env var, default 0.5
                        max_tokens=int(os.getenv("LLM_MAX_TOKENS", "1024")), # Use env var, default 1024
                        top_p=float(os.getenv("LLM_TOP_P", "0.9")), # Use env var, default 0.9
                        n_ctx=context_size, # Optimized context size
                        repeat_penalty=float(os.getenv("LLM_REPEAT_PENALTY", "1.1")), # Use env var, default 1.1
                        n_threads=num_threads, # Optimized thread count
                        n_batch=batch_size, # Optimized batch size
                        f16_kv=True, # Use half-precision for key/value cache (saves VRAM/RAM)
                        verbose=False, # Set to True for more LlamaCpp output
                        seed=int(os.getenv("LLM_SEED", "42")), # Use env var, default 42
                        # Stop sequences help the model know when to stop generating
                        # "Question:" is a good stop sequence for this prompt format
                        stop=["Question:"], # Removed "\n\n" to allow for paragraph breaks
                        streaming=True, # <--- ENABLE STREAMING HERE
                        last_n_tokens_size=int(os.getenv("LLM_LAST_N_TOKENS_SIZE", "64")), # Increased from 32 to 64
                        use_mlock=True, # Always use mlock for better performance
                        rope_freq_scale=float(os.getenv("LLM_ROPE_FREQ_SCALE", "0.5")), # Use env var, default 0.5
                        logits_all=False, # Usually not needed
                        n_gpu_layers=gpu_layers, # Number of layers to offload to GPU
                        use_mmap=True, # Always use mmap for better performance
                        offload_kqv=True if DEVICE == "cuda" else False, # Always offload KQV to GPU if CUDA available
                        tensor_split=None, # Auto tensor splitting
                        cache_capacity=2000 if DEVICE == "cuda" else None # Increased cache capacity for GPU
                    )

                    logger.info("Successfully initialized LLM with LlamaCpp (Streaming Enabled).")
                    return llm

                except ImportError:
                    logger.warning("LlamaCpp not installed. Please install it (`pip install llama-cpp-python`) for local model support.")
                    # Fall through to fallback options if LlamaCpp is not available
                except Exception as e:
                    logger.warning(f"Failed to initialize with LlamaCpp: {e}", exc_info=True)
                    # Fall through to fallback options on other LlamaCpp errors

            # --- API-based Model (HuggingFaceEndpoint) ---
            if MODEL_TYPE == "api":
                # Only use HuggingFaceEndpoint if we have a token
                if HF_TOKEN:
                    api_model_id = os.getenv("API_MODEL_ID", "HuggingFaceH4/zephyr-7b-beta")
                    logger.info(f"Using HuggingFaceEndpoint with model: {api_model_id}")
                    llm = HuggingFaceEndpoint(
                        repo_id=api_model_id,
                        task="text-generation", # Ensure task is correct for the model
                        max_new_tokens=int(os.getenv("LLM_MAX_TOKENS", "1024")), # Use env var
                        temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")), # Use env var
                        top_p=float(os.getenv("LLM_TOP_P", "0.95")), # Use env var
                        repetition_penalty=float(os.getenv("LLM_REPEAT_PENALTY", "1.15")), # Use env var
                        huggingfacehub_api_token=HF_TOKEN, # Pass token explicitly
                        # Streaming parameter might vary for HuggingFaceEndpoint
                        # Check HuggingFaceEndpoint documentation for streaming support
                        # For now, assuming it's not directly supported or handled differently
                        # streaming=True # <--- This might not work directly, check docs
                    )
                    logger.info("Successfully initialized LLM with HuggingFaceEndpoint.")
                    return llm
                else:
                    logger.error("API model type selected but no HUGGINGFACE_API_TOKEN provided in environment variables.")
                    # Fall through to fallback if token is missing

            # --- Fallback to HuggingFacePipeline (Transformers) ---
            # This fallback is useful if LlamaCpp or API models fail/are not configured.
            # Use a smaller model that is more likely to fit in limited VRAM.
            logger.info("Falling back to HuggingFacePipeline (Transformers)")
            try:
                # Use a model that's better suited for the GTX 1650's 4GB VRAM or CPU
                model_name = os.getenv("FALLBACK_MODEL_ID", "google/flan-t5-base") # Use env var, default to flan-t5-base
                logger.info(f"Loading fallback model {model_name} on device: {DEVICE}")

                # Configure model loading with appropriate settings for GPU/CPU
                # Use device_config for torch_dtype and device_index
                torch_dtype = device_config.get('torch_dtype', torch.float32) # Default to float32 if not in config
                device_index = device_config.get('device_index', -1) # Default to -1 (CPU) if not in config
                device_map = "auto" if DEVICE == "cuda" else None # Use device_map="auto" for GPU

                logger.info(f"Using dtype: {torch_dtype}, device_index: {device_index}, device_map: {device_map}")

                # Load tokenizer
                tokenizer = AutoTokenizer.from_pretrained(model_name)

                # Load model with optimized settings for GPU/CPU
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map=device_map, # Use device_map for automatic placement
                    torch_dtype=torch_dtype, # Use configured dtype
                    low_cpu_mem_usage=True if DEVICE == "cuda" else False # Helps when loading large models with GPU
                )

                # Create optimized pipeline
                pipe = pipeline(
                    "text-generation", # Or "text2text-generation" for T5 models
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens=int(os.getenv("LLM_MAX_TOKENS", "1024")), # Use env var
                    temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")), # Use env var
                    # device parameter in pipeline can be device_index or device string
                    # Using device_index from device_config
                    device=device_index
                    # Streaming might not be directly supported by HuggingFacePipeline
                    # Check pipeline documentation
                )

                llm = HuggingFacePipeline(pipeline=pipe)
                logger.info("Successfully initialized fallback model with HuggingFacePipeline.")
                return llm

            except Exception as fallback_error:
                logger.error(f"Error initializing fallback model: {fallback_error}", exc_info=True)
                # Fall through to final mock fallback

        except Exception as e:
            # Catch any exceptions during the LLM initialization process
            logger.error(f"An unexpected error occurred during LLM initialization: {e}", exc_info=True)
            # Final fallback - use a very simple mock LLM

        logger.error("All LLM initialization methods failed. Using a mock LLM.")
        class MockLLM:
             """A mock LLM to return an error message if all real LLMs fail."""
             def invoke(self, prompt: str, config: Optional[Dict] = None, **kwargs: Any) -> str:
                 logger.error("MockLLM.invoke called because real LLM failed to load.")
                 return "I'm sorry, but I couldn't load the language model. Please check your backend configuration and logs."
             # Add a stream method for compatibility
             def stream(self, prompt: str, config: Optional[Dict] = None, **kwargs: Any) -> Generator[str, None, None]:
                 logger.error("MockLLM.stream called because real LLM failed to load.")
                 yield "I'm sorry, but I couldn't load the language model. Please check your backend configuration and logs."


        return MockLLM()


    def _create_prompt_template(self):
        """
        Create the main prompt template for the RAG system.
        This template is used to format the context and query for the LLM.
        """
        # Template includes context, query, and an optional instruction with Markdown formatting
        template = """You are an advanced AI assistant similar to ChatGPT that provides comprehensive, natural-sounding answers based on document content. Your goal is to answer questions thoroughly while maintaining a conversational tone.

Context (document content):
{context}

Question: {query}

{instruction}

RESPONSE GUIDELINES:
1. Provide a comprehensive, detailed answer based on the document content
2. Write in a natural, conversational style like ChatGPT
3. Organize information logically and present it clearly
4. If the document doesn't contain enough information, acknowledge this politely
5. Stay factual and accurate to the document content
6. Don't mention "chunks," "context," or other technical details in your answer
7. Don't use phrases like "according to the context" or "based on the provided information"
8. Integrate information from different parts of the document seamlessly
9. For author information, only mention what is clearly stated in the document

Format your answer using Markdown for better readability:
- Use **bold** for important terms or concepts
- Use bullet points or numbered lists when appropriate
- Use headings (## or ###) for sections if needed
- Include clear paragraph breaks for readability

Answer:"""

        return PromptTemplate(
            input_variables=["context", "query", "instruction"],
            template=template
        )

    def _create_simple_prompt_template(self):
        """
        Create a simple prompt template for faster, more direct responses.
        Used when 'detailed' is False or in fast mode.
        """
        template = """You are an advanced AI assistant similar to ChatGPT. Answer the question based on the document content.

Document content:
{context}

Question: {query}

RESPONSE GUIDELINES:
- Provide a concise but comprehensive answer
- Write in a natural, conversational style like ChatGPT
- Be accurate and factual based on the document content
- Don't mention "chunks," "context," or other technical details
- Don't use phrases like "according to the document" or "based on the provided information"
- For author information, only mention what is clearly stated in the document

Format your answer using Markdown for better readability:
- Use **bold** for key points
- Use bullet points for lists when appropriate
- Keep paragraphs short and focused

Answer:"""

        return PromptTemplate(
            input_variables=["context", "query"],
            template=template
        )

    def _load_vector_stores(self, document_filter: Optional[List[str]] = None):
        """
        Load vector stores for all documents or filtered documents with caching.
        Vector stores are loaded from the EMBEDDINGS_DIR.
        """
        # Initialize cache if not already done
        if not hasattr(self, '_vector_stores_cache'):
            self._vector_stores_cache = {}

        # Create a cache key based on the document filter (sorted list of filenames)
        cache_key = 'all' if not document_filter else ','.join(sorted(document_filter))

        # Check if we have the combined vector stores for this filter in cache
        if cache_key in self._vector_stores_cache:
            logger.info(f"Using cached combined vector store for filter: {cache_key}")
            return self._vector_stores_cache[cache_key]

        # If not cached, load individual vector stores and combine them
        vector_stores = []
        logger.info(f"Attempting to load vector stores from {EMBEDDINGS_DIR}")

        # Check if the embeddings directory exists
        if not os.path.exists(EMBEDDINGS_DIR):
            logger.warning(f"Embeddings directory {EMBEDDINGS_DIR} does not exist. No documents processed?")
            return vector_stores # Return empty list if directory doesn't exist

        # Get all potential embedding directories (should correspond to processed documents)
        embedding_dirs = [d for d in os.listdir(EMBEDDINGS_DIR) if os.path.isdir(os.path.join(EMBEDDINGS_DIR, d))]
        logger.info(f"Found potential embedding directories: {embedding_dirs}")

        # Filter directories based on document_filter if provided
        if document_filter and document_filter != ["string"]: # Exclude default "string" value if passed
            # Extract base names without extensions for filtering (assuming dir names are filename bases)
            filter_bases = [os.path.splitext(doc)[0] for doc in document_filter]
            logger.info(f"Filtering embedding directories by document base names: {filter_bases}")
            embedding_dirs = [d for d in embedding_dirs if d in filter_bases]
            logger.info(f"Filtered embedding directories to load: {embedding_dirs}")
        else:
            logger.info("No document filter applied, attempting to load all available embedding directories.")

        # Load each vector store and combine them into a single searchable index
        combined_vector_store = None

        for dir_name in embedding_dirs:
            try:
                # Check if this individual store is cached
                store_cache_key = f"store_{dir_name}"
                if store_cache_key in self._vector_stores_cache:
                    logger.info(f"Using cached individual vector store: {dir_name}")
                    vector_store = self._vector_stores_cache[store_cache_key]
                else:
                    # Load from disk
                    vector_store_path = os.path.join(EMBEDDINGS_DIR, dir_name)
                    logger.info(f"Loading vector store from {vector_store_path}")
                    # Allow dangerous deserialization as we control the source
                    vector_store = FAISS.load_local(
                        vector_store_path,
                        self.embeddings, # Use the initialized embeddings model
                        allow_dangerous_deserialization=True
                    )
                    # Cache the individual store
                    self._vector_stores_cache[store_cache_key] = vector_store
                    logger.info(f"Successfully loaded and cached vector store: {dir_name}")

                # Combine with the main vector store
                if combined_vector_store is None:
                    combined_vector_store = vector_store
                else:
                    # Use the merge_from method to combine FAISS indices
                    logger.info(f"Merging vector store {dir_name}")
                    combined_vector_store.merge_from(vector_store)
                    logger.info("Merge successful.")

            except Exception as e:
                logger.error(f"Error loading or merging vector store {dir_name}: {e}", exc_info=True)
                # Continue loading other stores even if one fails

        # Cache the combined vector store for this filter combination
        if combined_vector_store is not None:
             self._vector_stores_cache[cache_key] = combined_vector_store
             logger.info(f"Cached combined vector store for filter: {cache_key}")

        # Return the combined vector store directly (or None if none were loaded)
        return combined_vector_store


    def query(
        self,
        query: str,
        document_filter: Optional[List[str]] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        detailed: bool = False,
        fast_mode: bool = False # Added fast_mode parameter
    ) -> Generator[Dict[str, Any], None, None]: # <--- Changed return type to Generator
        """
        Query the RAG system and stream the response.
        Retrieves relevant document chunks and uses an LLM to generate an answer,
        yielding chunks of the answer as they are generated.

        Args:
            query: The user's query string.
            document_filter: Optional list of document filenames to search within.
            max_tokens: Maximum number of tokens for the LLM to generate.
            temperature: Temperature for LLM text generation (controls randomness).
            detailed: If True, requests a more detailed answer from the LLM.
            fast_mode: If True, uses settings optimized for speed (fewer chunks).

        Yields:
            Dictionaries representing parts of the response:
            - {"type": "text", "content": "..."} for generated text chunks.
            - {"type": "metadata", "data": {...}} for sources and processing time (sent at the end).
            - {"type": "error", "message": "..."} if an error occurs.
        """
        start_time = time.time()
        logger.info(f"Starting streaming query process for: '{query}'")
        if document_filter:
            logger.info(f"Filtering by documents: {document_filter}")
        logger.info(f"Parameters: max_tokens={max_tokens}, temperature={temperature}, detailed={detailed}, fast_mode={fast_mode}")

        sources_for_frontend = [] # Initialize sources list

        try:
            # --- Determine search strategy and parameters ---
            combined_vector_store = self._load_vector_stores(document_filter)

            if not combined_vector_store:
                logger.warning("No vector stores available for querying.")
                yield {"type": "error", "message": "No documents have been processed yet. Please upload some documents first."}
                return # Exit generator

            # Query complexity and type detection
            query_lower = query.lower()
            query_word_count = len(query.split())

            # Keywords for different query types
            complex_keywords = ["explain", "detail", "how", "why", "compare", "difference", "analyze"]
            author_keywords = ["author", "authors", "who wrote", "contributors", "researchers"]
            
            # Check for author-related queries
            is_author_query = any(keyword in query_lower for keyword in author_keywords)
            
            # Complexity detection
            is_complex_query = (
                detailed or
                query_word_count > 7 or
                any(keyword in query_lower for keyword in complex_keywords)
            )

            # Optimize chunk retrieval based on query type - increased for better coverage
            if is_author_query:
                # For author queries, we need to retrieve more chunks to find author information
                # which is often at the beginning or end of documents
                k_value = 10  # Significantly increased for author queries
                max_chunks_for_context = 10
                logger.info("Author query detected: Retrieving up to 10 chunks to find author information.")
            elif fast_mode:
                k_value = 3  # Increased from 1 to 3 for better coverage even in fast mode
                max_chunks_for_context = 3
                logger.info("Fast mode enabled: Retrieving 3 chunks.")
            elif is_complex_query:
                k_value = 8  # Significantly increased for complex queries
                max_chunks_for_context = 8
                logger.info("Complex query detected: Retrieving up to 8 chunks.")
            else:
                k_value = 5  # Increased for better coverage
                max_chunks_for_context = 5
                logger.info("Simple query detected: Retrieving 5 chunks.")

            logger.info(f"Query complexity: {'Complex' if is_complex_query else 'Simple'}")
            logger.info(f"Retrieval parameters: k={k_value}, max_chunks_for_context={max_chunks_for_context}")

            # --- Retrieve relevant chunks ---
            all_relevant_chunks = []
            start_search_time = time.time()

            try:
                # Special handling for author queries
                if is_author_query:
                    logger.info("Using specialized search strategy for author information.")
                    
                    # First, try to find chunks specifically tagged as author information
                    try:
                        # Use metadata filtering to find chunks with author information
                        logger.info("Searching for chunks specifically tagged with author information")
                        author_chunks = combined_vector_store.similarity_search_with_score(
                            "author information affiliations correspondence",
                            k=3,
                            filter={"content_type": "author_info"}
                        )
                        
                        if author_chunks:
                            logger.info(f"Found {len(author_chunks)} chunks with author metadata")
                            relevant_chunks_with_scores = author_chunks
                        else:
                            # If no specific author chunks found, use a focused query
                            author_focused_query = f"authors affiliations correspondence {query}" 
                            logger.info(f"No author-tagged chunks found. Using focused query: '{author_focused_query}'")
                            relevant_chunks_with_scores = combined_vector_store.similarity_search_with_score(
                                author_focused_query,
                                k=k_value
                            )
                    except Exception as e:
                        logger.warning(f"Error during author-specific search: {e}. Falling back to standard search.")
                        # Fallback to standard search with author-focused query
                        author_focused_query = f"authors affiliations correspondence {query}"
                        relevant_chunks_with_scores = combined_vector_store.similarity_search_with_score(
                            author_focused_query,
                            k=k_value
                        )
                else:
                    # Standard search for non-author queries
                    logger.info(f"Searching combined vector store for relevant chunks (k={k_value}).")
                    relevant_chunks_with_scores = combined_vector_store.similarity_search_with_score(
                        query,
                        k=k_value
                    )
                
                all_relevant_chunks = [(chunk, score) for chunk, score in relevant_chunks_with_scores]
                all_relevant_chunks.sort(key=lambda x: x[1])
                top_chunks = [chunk for chunk, _ in all_relevant_chunks][:max_chunks_for_context]

                search_time = time.time() - start_search_time
                logger.info(f"Search completed in {search_time:.4f} seconds, found {len(top_chunks)} chunks for context.")

                if not top_chunks:
                    logger.warning("No relevant chunks found after search.")
                    yield {"type": "text", "content": "No relevant information found in the documents."}
                    # Still proceed to yield metadata chunk with empty sources
                    pass # Continue to context prep and generation (which will be based on empty context)

            except Exception as e:
                 logger.error(f"Error during vector store search: {e}", exc_info=True)
                 yield {"type": "error", "message": f"An error occurred during document search: {str(e)}"}
                 return # Exit generator


            # --- Prepare context for the LLM ---
            start_context_time = time.time()
            
            # Separate author chunks from content chunks for better organization
            author_chunks = []
            content_chunks = []
            
            for chunk in top_chunks:
                metadata = chunk.metadata if hasattr(chunk, 'metadata') else {}
                content_type = metadata.get('content_type', '')
                is_header = metadata.get('is_header', False)
                
                # Check if this chunk might contain author information
                chunk_lower = chunk.page_content.lower() if hasattr(chunk, 'page_content') else ''
                has_author_keywords = any(keyword in chunk_lower for keyword in 
                                         ["author", "authors", "correspondence", "affiliation", 
                                          "university", "institute", "department"])
                
                if content_type == 'author_info' or is_header or has_author_keywords:
                    author_chunks.append(chunk)
                else:
                    content_chunks.append(chunk)
            
            # If this is an author query, prioritize author chunks
            if is_author_query and author_chunks:
                logger.info(f"Author query detected with {len(author_chunks)} author chunks - prioritizing these")
                # Put author chunks first, then content chunks
                prioritized_chunks = author_chunks + content_chunks
            else:
                # For non-author queries, keep original order but still include author chunks first
                # for better context awareness
                if author_chunks and len(author_chunks) <= 2:  # Only if we have a reasonable number of author chunks
                    prioritized_chunks = author_chunks + [c for c in top_chunks if c not in author_chunks]
                else:
                    prioritized_chunks = top_chunks
            
            # Build context parts from prioritized chunks
            context_parts = []
            
            for i, chunk in enumerate(prioritized_chunks):
                metadata = chunk.metadata if hasattr(chunk, 'metadata') else {}
                source_path = metadata.get("source", "unknown_source")
                chunk_number = metadata.get("chunk", i + 1)
                content_type = metadata.get('content_type', '')
                is_header = metadata.get('is_header', False)

                # Extract filename from the source path
                original_filename = os.path.basename(source_path)
                
                # Add special marker for author information chunks
                chunk_type_marker = ""
                if content_type == 'author_info' or is_header:
                    chunk_type_marker = " [AUTHOR INFORMATION]"
                elif i < len(author_chunks) and chunk in author_chunks:
                    chunk_type_marker = " [LIKELY AUTHOR INFORMATION]"

                # Remove technical chunk details from the context sent to the LLM
                chunk_text = f"{chunk.page_content.strip()}\n"
                context_parts.append(chunk_text)

                # Prepare source information for the frontend
                source_info = {
                     "content": chunk.page_content.strip(),
                     "metadata": {
                         "original_filename": original_filename,
                         "chunk": chunk_number,
                         "source_path": source_path,
                         "content_type": content_type
                     },
                }
                sources_for_frontend.append(source_info)

            context = "\n\n".join(context_parts)
            context_prep_time = time.time() - start_context_time
            logger.info(f"Context prepared in {context_prep_time:.4f} seconds.")
            logger.debug(f"Context sent to LLM:\n{context}")


            # --- Generate answer using the LLM and stream ---
            start_generation_time = time.time()

            try:
                # Check for specific query types to customize formatting
                query_lower = query.lower()
                
                # Determine the type of formatting needed based on query keywords
                if "authors" in query_lower or "author" in query_lower:
                    instruction = """When discussing authors or contributors from the document:
- Present author information in a natural, conversational way
- Include author names, affiliations, and contact information exactly as they appear in the document
- Format author information in a clean, readable way using Markdown
- If you see author information with superscript numbers (like Author Name¹), include these appropriately
- Include the full institutional affiliations if provided in the document
- Include email addresses or other contact information if explicitly provided
- Do NOT make up or infer ANY biographical details, achievements, or other information
- If the document doesn't clearly list authors, acknowledge this politely
- If only partial author information is available, present what is known clearly
- Organize author information in a logical way (e.g., lead authors first, then co-authors)
- Present the information as if you were naturally explaining it to someone, not just listing it"""
                elif "list" in query_lower or "points" in query_lower or "main points" in query_lower:
                    instruction = """Provide your answer as a structured list with bullet points. Format each main point in **bold** and provide brief explanations.
Use Markdown formatting:
- Start with a brief introduction
- Use bullet points (- ) for each main point
- Bold (**) the key concepts
- Use sub-bullets for supporting details if needed"""
                elif "compare" in query_lower or "difference" in query_lower or "versus" in query_lower or "vs" in query_lower:
                    instruction = """Create a structured comparison using Markdown formatting:
- Use ### headings for each entity being compared
- Use bullet points to list characteristics
- Bold (**) the key differences
- Consider using a summary table if appropriate"""
                elif "step" in query_lower or "how to" in query_lower or "process" in query_lower or "procedure" in query_lower:
                    instruction = """Format your answer as a step-by-step guide:
1. Use numbered lists (1., 2., etc.) for sequential steps
2. Bold (**) the action in each step
3. Use clear, concise instructions
4. Add brief explanations where needed"""
                elif "short" in query_lower or "brief" in query_lower or "summary" in query_lower or "summarize" in query_lower:
                    instruction = """Provide a concise summary using Markdown:
- Keep your answer under 5 bullet points
- Focus only on the most essential information
- Bold (**) key terms
- Be direct and eliminate unnecessary details"""
                elif detailed:
                    instruction = """Provide a detailed, comprehensive answer based *only* on the provided context. 
Use Markdown formatting to enhance readability:
- Use ### headings to organize different sections
- Bold (**) important concepts and terms
- Use bullet points for lists of information
- Include specific details and examples from the text
- Create clear paragraph breaks between topics"""
                else:
                    instruction = """Provide a balanced answer with moderate detail.
Use Markdown formatting:
- Bold (**) key concepts
- Use bullet points where appropriate
- Create clear paragraph structure
- Highlight important terms with `backticks`"""
                
                # Use the appropriate prompt and streaming method
                if detailed or any(keyword in query_lower for keyword in ["list", "points", "compare", "difference", "step", "how to"]):
                    prompt = self.prompt_template.format(context=context, query=query, instruction=instruction)
                    logger.info("Using detailed prompt with custom formatting for streaming.")
                    # Use .stream() on the LLM
                    stream_iterator = self.llm.stream(prompt, config={"max_tokens": max_tokens, "temperature": temperature})
                else:
                    prompt = self.simple_prompt_template.format(context=context, query=query)
                    logger.info("Using simple prompt for streaming.")
                    # Use .stream() on the LLMChain
                    # LLMChain's stream method expects a dictionary matching prompt input variables
                    stream_iterator = self.simple_llm_chain.stream({"context": context, "query": query}, config={"max_tokens": max_tokens, "temperature": temperature})


                # Print the question to the console with clear formatting
                print("\n" + "="*80)
                print(f"Question: {query}")
                print("-"*80)
                print("Answer: ", end="", flush=True)  # Start the answer line
                
                # Yield text chunks from the stream
                answer_text = ""  # To collect the full answer
                for chunk in stream_iterator:
                    # LangChain's stream() yields different types depending on the LLM/Chain
                    # For LlamaCpp and simple text generation, it often yields strings.
                    # For chains, it might yield dictionaries. We need to handle both.
                    if isinstance(chunk, str):
                         text_chunk = chunk
                    elif isinstance(chunk, dict) and 'text' in chunk:
                         text_chunk = chunk['text']
                    else:
                         # If the chunk format is unexpected, log it and skip or handle
                         logger.warning(f"Received unexpected chunk format from stream: {chunk}")
                         continue # Skip this chunk

                    # Print the chunk to the console immediately
                    if text_chunk:  # Only print if there's actual text content
                        print(text_chunk, end="", flush=True)  # Print without newline and flush
                        answer_text += text_chunk  # Collect the full answer
                    
                    # Yield the text chunk with a type identifier
                    if text_chunk: # Only yield if there's actual text content
                        yield {"type": "text", "content": text_chunk}


                generation_time = time.time() - start_generation_time
                # Add a formatted completion message
                print("\n" + "-"*80)
                print(f"Generation completed in {generation_time:.2f} seconds")
                print("="*80 + "\n")
                logger.info(f"LLM generation completed in {generation_time:.4f} seconds.")


            except Exception as e:
                logger.error(f"Error during LLM streaming generation: {e}", exc_info=True)
                yield {"type": "error", "message": f"I encountered an error while generating the answer: {str(e)}"}
                # Clear sources if generation failed
                sources_for_frontend = []


        except Exception as e:
            # Catch any exceptions during the overall query process before streaming starts
            logger.error(f"An unexpected error occurred during query processing: {e}", exc_info=True)
            yield {"type": "error", "message": f"An unexpected error occurred during query processing: {str(e)}"}
            # Clear sources on overall error
            sources_for_frontend = []

        finally:
             # --- Send metadata (sources, processing time) after the stream ---
             total_processing_time = time.time() - start_time
             metadata_payload = {
                 "sources": sources_for_frontend,
                 "processing_time": total_processing_time,
                 "is_complex": is_complex_query
             }
             logger.info(f"Sending final metadata chunk: {metadata_payload}")
             # Yield the metadata as a JSON string chunk with a type identifier
             # The frontend will need to parse this specific chunk
             yield {"type": "metadata", "data": metadata_payload}
             logger.info("Streaming query process finished.")


# Dependency to get the RAGEngine instance
# This ensures the RAGEngine is initialized once and reused across requests
# (assuming your FastAPI app is run with a single worker or managed correctly)
_rag_engine_instance = None

def get_rag_engine() -> RAGEngine:
    """
    Dependency function to provide a singleton instance of RAGEngine.
    Initializes the engine on the first call.
    """
    global _rag_engine_instance
    if _rag_engine_instance is None:
        logger.info("Initializing RAGEngine instance...")
        _rag_engine_instance = RAGEngine()
        logger.info("RAGEngine instance initialized.")
    return _rag_engine_instance

