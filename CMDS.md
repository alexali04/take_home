``` shell -
python -m pytest regulation_task/tests/pre_processing/test_processor.py::test_doc_to_txt_processor

python regulation_task/correct.py


```

print(self.clean_text(doc[start_page].get_text("text")))