import argparse

def get_rag_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sop_model", type=str, default="claude-3-5-haiku-20241022")
    parser.add_argument("--chunking_model", type=str, default="claude-3-5-haiku-20241022")
    parser.add_argument("--logging_model", type=str, default="claude-3-5-sonnet-20241022")
    parser.add_argument("--sop_name", type=str, default="original.docx")
    parser.add_argument("--cut_off", action=argparse.BooleanOptionalAction)
    parser.add_argument("--cut_off_length", type=int, default=100)
    parser.add_argument("--max_tokens", type=int, default=8000)
    parser.add_argument("--data_dir", type=str, default="./regulation_task/data")

    parser.add_argument("--page_batch", type=int, default=10)
    parser.add_argument("--embedding_model", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--max_pdf_count", type=int, default=2)
    parser.add_argument("--use_llm_chunking", action=argparse.BooleanOptionalAction)

    # prompts
    parser.add_argument("--sys_prompt", type=str, default="compliance_expert_identity")
    parser.add_argument("--sop_prompt", type=str, default="chunking_sop")
    parser.add_argument("--reg_prompt", type=str, default="chunking_regulations")   # could also specify chunking_reg
    parser.add_argument("--logging_prompt", type=str, default="log_instructions") 
    parser.add_argument("--sys_prompt_logging", type=str, default="logging_sys_prompt")

    return parser

