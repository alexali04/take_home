import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import json

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
        self.id_to_clause = {}
        
    def add_document(self, document: str, id: str):
        """
        Add a document to the database.
        """
        embedding = self.embedder.embed_document(document)
        self.index.add(embedding)
        self.embeddings.append(embedding)
        self.ids.append(id)
        self.id_to_clause[id] = document

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
        Construct the database from a directory of regulatory jsons.
        """

        count = 0
        for file in os.listdir(directory):
            file_path = os.path.join(directory, file)
            clauses = json.load(open(file_path, "r"))["clauses"]
            for clause in clauses:
                clause = self.convert_to_str(clause)
                self.add_document(clause, count)
                count += 1
    
    def convert_to_str(self, clause):
        return f"{clause['title']} | {clause['text']} | Severity: {clause['severity']} | Consequence Level: {clause['consequence_level']} | Noncompliance: {clause['noncompliance_chance_level']}"

    def get_relevant_clauses(self, clause: str, k: int = 5) -> list[str]:
        """
        Get the k most relevant clauses from the database.
        """
        relevant_indices = self.search(clause, k)
        relevant_clauses = [self.id_to_clause[i] for i in relevant_indices]
        return relevant_clauses

    def get_relevant_clauses_from_path(self, clauses_path: str, k: int = 5) -> list[str]:
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
            relevant_indices = self.search(clause, k)
            relevant_clauses = [self.id_to_clause[i] for i in relevant_indices]
            results.append(relevant_clauses)

        return clauses, results
        

def construct_vector_database(args, json_dir: str):
    """
    Construct a vector database from the regulatory jsons.

    Specifically, for each regulatory text, chunk the text into clauses and add to database. 
    """
    embedder = Embedder(args.embedding_model)
    db = VectorDatabase(embedder)

    db.construct_database(json_dir)

    return db
    
