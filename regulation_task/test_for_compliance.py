import logging
from regulation_task.utils.parser import get_rag_parser
from regulation_task.document_processing.sop_processor import process_sop
from regulation_task.document_processing.reg_processor import extract_regulation_texts
from regulation_task.document_store.embedder import construct_vector_database


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Constants
DATA_DIR = "./regulation_task/data"


def main(args):
    data_dir = args.data_dir if args.data_dir != "" else DATA_DIR

    logging.info("Extracting regulatory texts...")
    regulation_texts_dir = extract_regulation_texts(args, data_dir)

    logging.info("Constructing vector database...")
    db = construct_vector_database(args, regulation_texts_dir)

    logging.info("Processing SOP...")
    clauses_path = process_sop(args, data_dir)

    logging.info("Getting relevant clauses...")
    relevant_clauses = db.get_relevant_clauses(clauses_path, k=5)

    logging.info("Generating log...")
    log = generate_log(args, clauses_path, relevant_clauses)
    logging.info(f"Log generated")

    print(log)



if __name__ == "__main__":
    parser = get_rag_parser()
    args = parser.parse_args()
    main(args)



















    