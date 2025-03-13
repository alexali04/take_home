from regulation_task.document_store.embedder import Embedder, VectorDatabase

def test_embedder():
    """
    Test the embedder class.
    """
    # breakpoint()
    embedder = Embedder()

    db = VectorDatabase(embedder)
    db.add_document("I love cats!", "1")
    db.add_document("I love dogs!", "2")
    db.add_document("I hate music!", "3")


    results = db.search("I love everyone!", k=2)
    assert len(results) == 2
    
