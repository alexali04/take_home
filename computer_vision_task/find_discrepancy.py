"""
Given SOP path and diagram path:
1. convert diagram pdf to images
2. extract symbols, text from images
3. find discrepancy between sop and diagram



"""


import os
from pathlib import Path
from utils.parser import get_sop_parser
from processor.pid_processor import pdf_to_img


def main(args):
    # pdf to jpeg images
    pid_pdf_path = f"{args.pid_data_dir}/{args.pid_path}"
    pid_png_path = f"{args.pid_data_dir}/images/"
    pdf_to_img(pid_pdf_path, pid_png_path)

    # group symbols / text from images






if __name__ == "__main__":
    parser = get_sop_parser()
    args = parser.parse_args()
    main(args)