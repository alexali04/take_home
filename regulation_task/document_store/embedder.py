import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
    
    def embed_document(self, document: str) -> np.ndarray:
        """
        Embed a document using the sentence transformer model.
        """
        embedding = self.model.encode(document)
        return np.array([embedding], dtype=np.float32)          # reshape to 2d array

    
class VectorDatabase:
    def __init__(self, embedder: Embedder = None):
        self.embedder = embedder
        self.embedding_dim = self.embedder.embedding_dim
        self.index = faiss.IndexFlatL2(self.embedding_dim) # search via cosine similarity
        self.embeddings = []
        self.ids = []
        
    def add_document(self, document: str, id: str):
        """
        Add a document to the database.
        """
        embedding = self.embedder.embed_document(document)
        self.index.add(embedding)
        self.embeddings.append(embedding)
        self.ids.append(id)

    def search(self, query: str, k: int = 10) -> list[str]:
        """
        Search the database for the k most similar documents to the query.
        """
        embedding = self.embedder.embed_document(query)
        _, indices = self.index.search(embedding, k)
        indices = indices.reshape(-1)
        return [self.ids[i] for i in indices]

