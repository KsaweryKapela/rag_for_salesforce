from config import DB_NAME
from db.db_utils import ChromaDB

chroma_db = ChromaDB(collection_name=DB_NAME)