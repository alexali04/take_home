from regulation_task.pre_processor.sop_processor import SOP_Processor, Regulatory_API_Prompt

def test_sop_processor():
    sop_processor = SOP_Processor("./regulation_task/data/sop/original.docx")
    prompt = Regulatory_API_Prompt(
        sys_prompt="You must say hello back! Only say 'hello'! No preamble!",
        user_prompt="", 
        max_tokens=10
    )

    response = sop_processor.extract_clauses(prompt, context="hello!")
    assert response == "hello"



