``` shell -
python -m pytest regulation_task/tests/pre_processing/test_processor.py::test_sop_processor
python -m pytest regulation_task/tests/pre_processing/test_processor.py::test_doc_to_txt_processor
python -m pytest regulation_task/tests/document_store/test_faiss.py::test_embedder
python -m pytest regulation_task/tests/test_pipeline.py::test_pipeline


python regulation_task/tests/document_store/test_faiss.py
python regulation_task/tests/test_pipeline.py


python regulation_task/test_for_compliance.py --use_llm_chunking

find regulation_task/data/regulatory_texts -type f -exec wc -c {} + | awk '{ total += $1 } END { print total }'
```

10 documents - 3.8 million tokens
-- claude haiku - spends roughly $3 on extracting / cleaning this
-- too many API calls? call for document extractions on SOP, document extraction on 

