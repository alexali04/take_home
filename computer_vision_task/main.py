import os
from utils.parser import get_sop_parser
from processor.pid_processor import analyze_pid

def main(args):
    sop_data_dir = args.sop_data_dir
    pid_data_dir = args.pid_data_dir
    ref_data_dir = args.ref_data_dir

    sop_path = os.path.join(sop_data_dir, args.sop_path)
    pid_path = os.path.join(pid_data_dir, args.pid_path)
    arrowhead_template_path = os.path.join(sop_data_dir, args.arrowhead_template_path)


    analyze_pid(pid_path)









if __name__ == "__main__":
    parser = get_sop_parser()
    args = parser.parse_args()
    main(args)