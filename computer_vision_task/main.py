import os
from utils.parser import get_sop_parser
from processor.pid_processor import PIDAnalyzer

def main(args):
    sop_data_dir = args.sop_data_dir
    pid_data_dir = args.pid_data_dir
    ref_data_dir = args.ref_data_dir

    sop_path = os.path.join(sop_data_dir, args.sop_path)
    pid_path = os.path.join(pid_data_dir, args.pid_path)
    arrowhead_template_path = os.path.join(sop_data_dir, args.arrowhead_template_path)

    # 1. parse pdf

    processor = PIDAnalyzer(
        arrowhead_template_path=arrowhead_template_path,
        detection_model=args.detection_model,
        text_detection_threshold=args.text_detection_threshold,
        arrow_detection_threshold=args.arrow_detection_threshold
    )

    components = processor.detect_components(pid_path)
    text_data = processor.extract_text_data(pid_path)
    arrow_heads = processor.detect_arrow_heads(pid_path)
    processor.build_pid_graph(text_data, arrow_heads, components)









if __name__ == "__main__":
    parser = get_sop_parser()
    args = parser.parse_args()
    main(args)