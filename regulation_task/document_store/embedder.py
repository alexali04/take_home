import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os

from utils.prompting import Regulatory_API_Prompt

class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

    def embed_document(self, document: str) -> np.ndarray:
        """
        Embed a document using the sentence transformer model.
        """
        embedding = self.model.encode(document)
        return np.array([embedding], dtype=np.float32)    
              # reshape to 2d array

    
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
    
    def construct_database(self, directory: str):
        """
        Construct the database from a directory of documents.
        """
        for file in os.listdir(directory):
            with open(os.path.join(directory, file), "r") as f:
                # need to chunk it into regulatory clauses
                clauses = self.chunk_to_regulatory_clauses(f.read())
                for clause in clauses:
                    self.add_document(clause, file)
    
    def get_relevant_clauses(self, clauses_path: str, k: int = 5) -> list[str]:
        """
        Get the k most relevant clauses from the database.

        Args:
            clauses_path (str): The path to the file containing the SOP clauses.
            k (int): The number of relevant clauses to return per clause. 

        Returns:
            list[list[str]]: The k most relevant clauses for each clause in the SOP.
        """

        results = []
        clauses = open(clauses_path, "r").read()
        for clause in clauses:

            results.append(self.search(clause, k))
        
        assert len(results) == len(clauses)
        assert len(results[0]) == k
        return results
    
    def chunk_to_regulatory_clauses(document: str, use_llm: bool = False) -> list[str]:
        """
        Chunk a regulatory document into clauses.

        WARNING: LLM chunking is EXPENSIVE! Only use if data is highly unstructured. 
        """
        if not use_llm:
            return document.split("\n\n")
    
        # Use LLM to chunk the document into regulatory clauses
 

def construct_vector_database(args, regulatory_texts_dir: str):
    """
    Construct a vector database from the regulatory texts.

    Specifically, for each regulatory text, chunk the text into clauses and add to database. 
    """
    embedder = Embedder(args.embedding_model)
    db = VectorDatabase(embedder)

    for file in os.listdir(regulatory_texts_dir):
        file_path = os.path.join(regulatory_texts_dir, file)
        with open(file_path, "r") as f:
            clauses = f.read()
            for clause in clauses:
                db.add_document(clause, file_path)

    return db
    
