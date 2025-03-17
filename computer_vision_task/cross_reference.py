"""
Given SOP path and diagram path:
1. convert diagram pdf to images
2. extract symbols, text from images
3. find discrepancy between sop and diagram
"""

from cv_utils.parser import get_sop_parser
from processor.pid_processor import pdf_to_img
from symbol_extraction.recog_symbol import construct_graphs_for_dir
from processor.sop_processor import get_discrepancies, Regulatory_API_Prompt, read_sop


def main(args):
    # pdf to jpeg images
    pid_pdf_path = f"{args.pid_data_dir}/{args.pid_path}"
    pid_img_path = f"{args.pid_data_dir}/images"
    pdf_to_img(pid_pdf_path, pid_img_path)

    # group symbols / text from images
    weights_path = "./computer_vision_task/symbol_extraction/best.pt"
    graphs = construct_graphs_for_dir(pid_img_path, show_img=False, weight_path=weights_path)

    # find discrepancy between sop and diagram
    # 1st, we find instructions for the LLM
    instructions = open(f"{args.sop_data_dir}/{args.model_prompt}", "r").read()

    # 2nd, we read the SOP
    sop = read_sop(f"{args.sop_data_dir}/{args.sop_path}")

    instructions += f"\nSOP: {sop}"

    api_prompt = Regulatory_API_Prompt(
            model=args.model,
            sys_prompt="You are a regulatory compliance expert. You are given a standard operating procedure (SOP) and a P&ID diagram. Your job is to find the discrepancy between the SOP and the P&ID diagram.",
            content=instructions,
            max_tokens=1000
        )

    for graph in graphs:
        discrepancies = get_discrepancies(api_prompt, str(graph))
        api_prompt.reset_content()
        print(api_prompt.content[-500:])
        print(discrepancies)
        

if __name__ == "__main__":
    parser = get_sop_parser()
    args = parser.parse_args()
    main(args)

