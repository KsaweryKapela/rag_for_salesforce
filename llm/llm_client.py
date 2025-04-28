from google import genai
from config import GEMINI_API_KEY
import chromadb.utils.embedding_functions as embedding_functions

google_client = genai.Client(api_key=GEMINI_API_KEY)
google_ef = embedding_functions.GoogleGenerativeAiEmbeddingFunction(api_key=GEMINI_API_KEY,
                                                                    model_name="text-embedding-004",
                                                                    task_type="RETRIEVAL_DOCUMENT")
