import argparse


def get_sop_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sop_path", type=str, default="sop.docx")
    parser.add_argument("--pid_path", type=str, default="diagram.pdf")

    parser.add_argument("--model_prompt", type=str, default="sop_vs_graph.txt")
    parser.add_argument("--model", type=str, default="claude-3-5-haiku-20241022")
    parser.add_argument("--sop_data_dir", type=str, default="./computer_vision_task/data/sop")
    parser.add_argument("--pid_data_dir", type=str, default="./computer_vision_task/data/p&id")
    return parser
