Currently choosing to do computer vision task and regulation task. 


## Standard Operating Procedure

Goal: build tool that processes SOP w/ regulatory documents.

First step is EDA. 

A SOP is a essentially an instruction book for how to perform a specific task consistently and safely. From a quick scan, we can see that this document has to do with purging condensation by periodically opening a valve. 

Regulatory documents contain additional information that may be relevant to completing the task in a manner consistent with legal guidelines. These regulatory documents seem to be mostly text-based data contained in pdfs. That being said, they're of an inconsistent format (REG-ANSI A92.2), and have various diagrams scattered about. 

Second step is planning. 

So the problem is that there may be inconsistencies between the SOP and the regulatory documents. At a macro-level:

1. We want to be able to search through the space of regulatory documents and figure out which clauses are relevant to the document. 

2. Then, we output a log, detailing the discrepancies. 

Optional additional step: Annotate the SOP somehow. 

At a more detailed level, we almost certainly want to use some sort of RAG. So, we embed the SOP into query vectors. Then, we can perform some search in some vector database to come up with errors. 

Main anticipated difficulty has to do with processing the regulation data. This data is highly unstructured - maybe some combination of OCR + figure recognition?

Plan (Interview):
1. Document pre-processing
a. Convert SOP into text
b. Using OCR + other PDF parsers, convert regulatory PDFs into text (in real scenario, would search for multi-model embedding strategy)
c. LLM-based extraction - use LLM to chunk texts into regulatory clauses. 
    (i) - interesting questions on how to best prompt LLM for extraction

2. RAG-LLM
a. embed SOP chunks as query vectors
b. embed regulatory texts as key vector
c. prompt LLM to fix SOP with most relevant texts

Additional Steps:
1. training code for transformer - constructs better multi-modal embeddings via attn mechanism
2. optional front-end website









