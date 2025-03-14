import os
import fitz 
import re
from typing import Optional


class Doc_Processor:
    def __init__(self, pdf_directory="./regulation_task/data/regulations", output_directory="./regulation_task/data/regulatory_texts"):
        """
        Initialize the PDFParser with a directory of PDFs.
        args:
           (str) pdf_directory: Path to the directory containing PDFs.
           (str) output_directory: Path to store extracted text files.
        """
        self.pdf_directory = pdf_directory
        self.output_directory = output_directory
        os.makedirs(self.output_directory, exist_ok=True)
    

    def clean_text(self, text):
        """
        Cleans extracted text by removing boilerplate, fixing formatting, and removing hyphenated words.
        """
        # Remove page numbers and metadata (e.g., "PO 00000 Frm 00567 Fmt 8010 Sfmt 8010")
        text = re.sub(r"PO\s+\d{5}\s+Frm\s+\d+\s+Fmt\s+\d+\s+Sfmt\s+\d+", "", text)

        # Remove Editorial Notes, List of CFR Sections Affected, and Federal Register citations
        text = re.sub(r"EDITORIAL NOTE:.*?(www\.\S+)?", "", text, flags=re.DOTALL)
        text = re.sub(r"List of CFR Sections Affected.*?(www\.\S+)?", "", text, flags=re.DOTALL)
        text = re.sub(r"FEDERAL REGISTER citations.*?(www\.\S+)?", "", text, flags=re.DOTALL)

        # Remove references like "AUTHORITY: 49 U.S.C. 106(g), 40103, ..."
        text = re.sub(r"AUTHORITY:.*", "", text)
        text = re.sub(r"SOURCE:.*", "", text)
        text = re.sub(r"EFFECTIVE DATE NOTE:.*", "", text)

        # Remove "Reserved" sections
        text = re.sub(r"PART\s+\d+\s+\[RESERVED\]", "", text)

        # Remove hyphenation (split words across lines like "air- space" -> "airspace")
        text = re.sub(r"(\w+)-\s+(\w+)", r"\1\2", text)

        text = re.sub(r"\n\s*\n", "\n", text).strip()
        return text
    

    def parse_pdf(self, pdf_path, page_batch=10, remove_existing=True):
        pdf_name = os.path.basename(pdf_path).replace(".pdf", "")
        output_file = os.path.join(self.output_directory, f"{pdf_name}.txt")

        if os.path.exists(output_file) and remove_existing:
            os.remove(output_file)

        try:
            doc = fitz.open(pdf_path)
            total_pages = len(doc)

            with open(output_file, "w", encoding="utf-8") as f:
                for start_page in range(0, total_pages, page_batch):
                    end_page = min(start_page + page_batch, total_pages)
                    text_batch = [self.clean_text(doc[page].get_text("text")) for page in range(start_page, end_page)]

                    f.write("\n".join(text_batch) + "\n\n")

            print(f"Extracted text saved to: {output_file}")
            return output_file
        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")


    def parse_directory_of_pdfs(self, page_batch=5, max_pdfs: Optional[int] = None):
        """
        Parse all PDFs in the directory.
        """
        pdf_files = [f for f in os.listdir(self.pdf_directory) if f.endswith(".pdf")]
        if not pdf_files:
            print("No PDF files found in the directory.")
            return
        
        output_files = []
        if max_pdfs is not None:
            pdf_files = pdf_files[:max_pdfs]   # Only process the first max_pdfs PDFs


        for pdf in pdf_files:
            pdf_path = os.path.join(self.pdf_directory, pdf)
            print(f"Processing {pdf}...")
            self.parse_pdf(pdf_path, page_batch=page_batch)
            
        return output_files



def extract_regulation_texts(args, data_dir: str):
    """
    Process all PDFs in the regulations directory and save the extracted text to the regulatory_texts directory.
    """

    reg_dir = os.path.join(data_dir, "regulations")
    regulation_texts_dir = os.path.join(data_dir, "regulatory_texts")

    document_processor = Doc_Processor(
        pdf_directory=reg_dir,
        output_directory=regulation_texts_dir
    )

    document_processor.parse_directory_of_pdfs(
        page_batch=args.page_batch
    )

    return regulation_texts_dir
    