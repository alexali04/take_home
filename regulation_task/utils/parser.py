import argparse

def get_rag_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sop_path", type=str, default="./regulation_task/data/sop/original.docx")
    parser.add_argument("--cut_off", type=bool, default=False)
    parser.add_argument("--cut_off_length", type=int, default=100)
    parser.add_argument("--sys_prompt_path", type=str, default="regulatory_compliance_expert.txt")
    return parser

