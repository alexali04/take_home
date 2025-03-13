from regulation_task.utils.parser import get_rag_parser
from regulation_task.document_processing.sop_processor import SOP_Processor, Regulatory_API_Prompt



def main(args):

    # 1. Process SOP
    sop_processor = SOP_Processor(args.sop_path)

    sys_prompt = open(f"./regulation_task/data/prompts/{args.sys_prompt_text}.txt").read()
    extraction_prompt = open(f"./regulation_task/data/prompts/{args.extraction_prompt_text}.txt").read()

    prompt = Regulatory_API_Prompt(
        model=args.model,
        sys_prompt=sys_prompt,
        assistant_prompt=None,
        content=extraction_prompt,
        max_tokens=2000
    )
    
    clause_dict = sop_processor.extract_clauses(prompt)

    # 2. Process Regulatory Clauses

    # 3. Convert text objects to vector database embeddings

    # 4. Query vector database for relevant clauses

    # 5. Feed query + relevant keys to LLM

    # 6. Return corrected clauses
    






if __name__ == "__main__":
    parser = get_rag_parser()
    args = parser.parse_args()
    main(args)



















    