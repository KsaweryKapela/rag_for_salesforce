import os
from dotenv import load_dotenv

load_dotenv()

DOCS_PATH = os.getenv("DOCS_PATH")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DB_NAME = os.getenv("DB_NAME", "salesforcedocretrival")
DB_PATH = os.getenv("DB_PATH", "./chroma")
MODEL_ID = os.getenv("MODEL_ID", "gemini-2.5-flash-preview-04-17")

