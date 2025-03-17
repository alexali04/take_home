"""
Given SOP path and diagram path:
1. convert diagram pdf to images
2. extract symbols, text from images
3. find discrepancy between sop and diagram
"""

from cv_utils.parser import get_sop_parser
from processor.pid_processor import pdf_to_img
from symbol_extraction.recog_symbol import detect_symbols_text_for_dir

def main(args):
    # pdf to jpeg images
    pid_pdf_path = f"{args.pid_data_dir}/{args.pid_path}"
    pid_img_path = f"{args.pid_data_dir}/images"
    pdf_to_img(pid_pdf_path, pid_img_path)


    # group symbols / text from images
    weights_path = "./computer_vision_task/symbol_extraction/best.pt"
    detect_symbols_text_for_dir(pid_img_path, show_img=True, weight_path=weights_path)





if __name__ == "__main__":
    parser = get_sop_parser()
    args = parser.parse_args()
    main(args)

