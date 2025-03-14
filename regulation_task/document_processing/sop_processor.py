"""
Nice reference: https://docs.anthropic.com/en/docs/about-claude/use-case-guides/legal-summarization

https://github.com/anthropics/anthropic-cookbook/blob/4a7be656bd1a6e2149f9f1c40678dac379a857a7/skills/summarization/guide.ipynb

Options:

summarization
---- prompt model to summarize the SOP
---- disadvantage: lossy summary may lose key legal details

few-shot summarization
---- we prompt the model w/ examples of documents + regulatory clauses
---- helps the model "understand" the task
---- disadvantage: longer context lengths, may infer "bad patterns"

guided summarization (chosen)
---- provide the model w/ detailed instructions

practically, domain guided summarization
---- if we know we're dealing with a warehouse compliance doc, we'd prompt it w/ specific terminology
---- not chosen due to time constraint
"""


import docx
import json
import os

from regulation_task.utils.prompt_utils import Regulatory_API_Prompt, extract_clauses_from_docx

class SOP_Processor():
    """
    Processes and chunks SOP. 
    """
    def __init__(
        self, 
        sop_path: str = "./regulation_task/data/sop/original.docx",
        cut_off: bool = False, 
        cut_off_length: int = 100
    ):
        assert sop_path.endswith(".docx"), "SOP must be a .docx file"

        self.sop_path = sop_path
        self.cut_off = cut_off
        self.cut_off_length = cut_off_length
    
    def manual_para_chunking(self):
        f = open(self.sop_path, "rb")
        doc = docx.Document(f)

        paras = []
        curr_para = ""
        in_para = False

        for para in doc.paragraphs: 
            if self.cut_off and len(curr_para) > self.cut_off_length:
                curr_paras = [curr_para[:self.cut_off_length], curr_para[self.cut_off_length:]]
                paras.extend(curr_paras)
                curr_para = ""

            if para.text.strip() == "": 
                if not in_para:
                    continue 
                else: # we are in a paragraph and we hit an empty line --> we are done with the paragraph
                    in_para = False
                    paras.append(curr_para)
                    curr_para = ""

            else: # not empty line --> we are in a paragraph
                if not in_para: 
                    in_para = True
                curr_para += para.text + "\n"
            
        return paras

    def get_text_from_docx(self):
        f = open(self.sop_path, "rb")
        doc = docx.Document(f)
        return "\n".join([para.text for para in doc.paragraphs])

    def post_process_clauses(self, clause_str: str):
        """Returns json object from string"""
        clauses = json.loads(clause_str)
        return clauses

def process_sop(args, data_dir: str):
    sop_path = os.path.join(data_dir, "sop", args.sop_name)
    prompt_dir = os.path.join(data_dir, "prompts")

    sop_processor = SOP_Processor(
        sop_path=sop_path,
        cut_off=args.cut_off,
        cut_off_length=args.cut_off_length
    )
    
    sys_prompt = open(f"{prompt_dir}/{args.sys_prompt}.txt").read()
    sop_prompt = open(f"{prompt_dir}/{args.sop_prompt}.txt").read()

    api_prompt = Regulatory_API_Prompt(
        model=args.sop_model,
        sys_prompt=sys_prompt,
        content=sop_prompt,
        max_tokens=8000
    )

    context = sop_processor.get_text_from_docx()

    clause_text = extract_clauses_from_docx(api_prompt, context)

    clause_dict = sop_processor.post_process_clauses(clause_text)

    clauses_path = os.path.join(data_dir, "sop", f"sop_clauses.json")
    json.dump(clause_dict, open(clauses_path, "w"), indent=4)

    return clauses_path

    
    







    
