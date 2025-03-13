Implements embedding model + vector database 

Given a list of text strings representing text data taken from regulatory clauses PDFs, we can chunk the text
and pass it into a vector database. This folder will also implement querying for finding relevant clauses given some SOP. 

Considerations:
- Chunk size should align with regulatory clause 
-- in real life, would use a cheap model to extract regulatory clauses from - is this scalable? 200 page pdf isn't cheap to process + many output tokens

instead, we'll handle manual chunking

