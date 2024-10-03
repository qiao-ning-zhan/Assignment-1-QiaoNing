from typing import Optional, List, Dict
import chromadb
from chromadb.api.types import EmbeddingFunction
from chromadb.config import Settings
import os
from uuid import uuid4
from dotenv import load_dotenv

load_dotenv()

class OpenAIEmbedder(EmbeddingFunction):
    def __init__(self):
        from openai import OpenAI
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = "text-embedding-3-large"

    def __call__(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.client.embeddings.create(input=texts, model=self.model).data
        return [e.embedding for e in embeddings]

class VectorDBManager:
    def __init__(self, database_name: str):
        self.database_name = database_name
        self.client = chromadb.PersistentClient(
            path=f"./data/vector_storage/{database_name}",
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name=database_name,
            embedding_function=OpenAIEmbedder(),
            metadata={"hnsw:space": "cosine"}
        )

    def insert_document(self, content: str, document_id: Optional[str] = None) -> str:
        if document_id is None:
            document_id = str(uuid4())
        
        self.collection.add(
            documents=[content],
            metadatas=[{"source": self.database_name}],
            ids=[document_id]
        )
        return document_id

    def search_similar(self, query: str, limit: int) -> str:
        results = self.collection.query(
            query_texts=[query],
            n_results=limit,
            include=["documents", "metadatas"]
        )
        return results['documents'][0][0] if results['documents'] else ""

    def modify_document(self, document_id: str, new_content: str) -> None:
        self.collection.update(
            ids=[document_id],
            documents=[new_content],
            metadatas=[{"source": self.database_name}]
        )

    def remove_document(self, document_id: str) -> None:
        self.collection.delete(ids=[document_id])