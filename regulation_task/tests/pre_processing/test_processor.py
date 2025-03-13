import os
from regulation_task.document_processing.sop_processor import SOP_Processor, Regulatory_API_Prompt
from regulation_task.document_processing.reg_processor import Doc_Processor

def test_sop_processor():
    """
    Ensures API call is working
    """
    sop_processor = SOP_Processor("./regulation_task/data/sop/original.docx")
    prompt = Regulatory_API_Prompt(
        sys_prompt="You must say hello back! Only say 'hello'! No preamble!",
        content="", 
        max_tokens=10
    )

    response = sop_processor.extract_clauses_from_docx(prompt, context="hello!")
    assert response == "hello"


def test_doc_to_txt_processor():
    """
    Ensures PDF parses and is converted to txt
    """
    doc_processor = Doc_Processor()
    doc_processor.parse_pdf("./regulation_task/data/regulations/REG-14 CFR Part 77.pdf", remove_existing=True)
    assert os.path.exists("./regulation_task/data/regulatory_texts/REG-14 CFR Part 77.txt")
    



