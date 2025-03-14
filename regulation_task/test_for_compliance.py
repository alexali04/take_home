import logging
import json
from regulation_task.utils.parser import get_rag_parser
from regulation_task.utils.prompt_utils import generate_log
from regulation_task.document_processing.sop_processor import process_sop
from regulation_task.document_processing.reg_processor import extract_regulation_clauses
from regulation_task.document_store.embedder import construct_vector_database


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Constants
DATA_DIR = "./regulation_task/data"


def main(args):
    data_dir = args.data_dir if args.data_dir != "" else DATA_DIR

    logging.info("Extracting regulatory clauses into jsons...")
    regulatory_json_dir = extract_regulation_clauses(args, data_dir)

    logging.info("Processing SOP...")
    clauses_path = process_sop(args, data_dir)

    logging.info("Constructing vector database...")
    db = construct_vector_database(args, regulatory_json_dir)

    logging.info("Getting relevant clauses...")
    sop_clauses, regulatory_clauses = db.get_relevant_clauses_from_path(clauses_path, k=3)

    logging.info(f"Number of SOP clauses: {len(sop_clauses)}")
    logging.info(f"Number of regulatory clauses: {len(regulatory_clauses)}")
    logging.info(f"Number of regulatory clause contextual retrievals per SOP clause: {len(regulatory_clauses[0])}")



    # sop_clauses: list of strings (key vectors)
    # regulatory_clauses: list of list of 5 strings (query vectors)

    logging.info("Generating log...")

    log = generate_log(args, sop_clauses, regulatory_clauses, data_dir)

    print(log)

    logging.info(f"Number of violations: {len(log)}")




if __name__ == "__main__":
    parser = get_rag_parser()
    args = parser.parse_args()
    main(args)



















    