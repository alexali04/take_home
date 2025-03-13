import os
from regulation_task.utils.parser import get_rag_parser
from regulation_task.document_processing.sop_processor import SOP_Processor, Regulatory_API_Prompt
from regulation_task.document_processing.reg_processor import Doc_Processor


# Constants
DATA_DIR = "./regulation_task/data"

def main(args):
    breakpoint()
    data_dir = args.data_dir if args.data_dir != "" else DATA_DIR

    sop_path = os.path.join(data_dir, "sop", args.sop_name)
    reg_dir = os.path.join(data_dir, "regulations")
    prompt_dir = os.path.join(data_dir, "prompts")
    regulation_texts_dir = os.path.join(data_dir, "regulatory_texts")

    # 2. Process SOP
    sop_processor = SOP_Processor(
        sop_path=sop_path,
        cut_off=args.cut_off,
        cut_off_length=args.cut_off_length
    )

    sys_prompt = open(f"{prompt_dir}/{args.sys_prompt_text}.txt").read()
    extraction_prompt = open(f"{prompt_dir}/{args.extraction_prompt_text}.txt").read()

    prompt = Regulatory_API_Prompt(
        model=args.model,
        sys_prompt=sys_prompt,
        assistant_prompt=None,
        content=extraction_prompt,
        max_tokens=2000
    )
    
    clause_dict = sop_processor.extract_clauses_from_docx(prompt)

    # 3. Process Regulatory Texts
    document_processor = Doc_Processor(
        pdf_directory=reg_dir,
        output_directory=regulation_texts_dir
    )

    document_processor.parse_directory_of_pdfs(
        page_batch=args.page_batch
    )

    # 4. Construct vector database
    




    # 3. Convert text objects to vector database embeddings

    # 4. Query vector database for relevant clauses

    # 5. Feed query + relevant keys to LLM

    # 6. Return corrected clauses
    






if __name__ == "__main__":
    parser = get_rag_parser()
    args = parser.parse_args()
    main(args)



















    