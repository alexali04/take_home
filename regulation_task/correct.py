from regulation_task.utils.parser import get_rag_parser
from regulation_task.pre_processor.sop_processor import SOP_Processor
from regulation_task.pre_processor.sop_processor import Regulatory_API_Prompt



def main(args):
    sop_processor = SOP_Processor(args.sop_path)

    sys_prompt = open(f"./regulation_task/data/prompts/{args.sys_prompt_path}").read()

    prompt = Regulatory_API_Prompt(
        model=args.model,
        sys_prompt=sys_prompt,
        assistant_prompt=None,
        content="",
    )
    

    clauses = sop_processor.extract_clauses(prompt)
    breakpoint()
    






if __name__ == "__main__":
    parser = get_rag_parser()
    args = parser.parse_args()
    main(args)



















    