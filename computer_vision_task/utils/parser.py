import argparse


def get_sop_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sop_path", type=str, default="original.docx")
    parser.add_argument("--pid_path", type=str, default="pid.png")
    parser.add_argument("--arrowhead_template_path", type=str, default="arrowhead.png")
    parser.add_argument("--detection_model", type=str, default="yolo")
    parser.add_argument("--sop_prompt", type=str, default="")
    parser.add_argument("--sop_data_dir", type=str, default="./data/sop")
    parser.add_argument("--pid_data_dir", type=str, default="./data/pid")
    parser.add_argument("--ref_data_dir", type=str, default="./data/ref")
    parser.add_argument("--text_detection_threshold", type=int, default=50)
    parser.add_argument("--arrow_detection_threshold", type=float, default=0.8)
    return parser
