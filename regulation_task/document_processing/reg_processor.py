import os
import fitz # PyMuPDF
import PyPDF2

class DocProcessor:
    def __init__(self, pdf_directory, output_directory="parsed_regulatory_texts"):
        """
        Initialize the PDFParser with a directory of PDFs.
        :param pdf_directory: Path to the directory containing PDFs.
        :param output_directory: Path to store extracted text files.
        """
        self.pdf_directory = pdf_directory
        self.output_directory = output_directory
        os.makedirs(self.output_directory, exist_ok=True)
    

    def parse_pdf(self, pdf_path, page_batch=10):
        pdf_name = os.path.basename(pdf_path).replace(".pdf", "")
        output_file = os.path.join(self.output_directory, f"{pdf_name}.txt")

        try:
            doc = fitz.open(pdf_path)
            total_pages = len(doc)

            with open(output_file, "w", encoding="utf-8") as f:
                for start_page in range(0, total_pages, page_batch):
                    end_page = min(start_page + page_batch, total_pages)
                    text_batch = [doc[page].get_text("text") for page in range(start_page, end_page)]
                    f.write("\n".join(text_batch) + "\n\n")

            print(f"Extracted text saved to: {output_file}")
        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")
    

    def parse_directory(self, page_batch=5):
        """
        Parse all PDFs in the directory.
        :param page_batch: Number of pages to batch together while extracting text.
        """
        pdf_files = [f for f in os.listdir(self.pdf_directory) if f.endswith(".pdf")]
        if not pdf_files:
            print("No PDF files found in the directory.")
            return
        
        for pdf in pdf_files:
            pdf_path = os.path.join(self.pdf_directory, pdf)
            print(f"Processing {pdf}...")
            self.parse_pdf(pdf_path, page_batch=page_batch)
            break

            
pdf_directory = "./regulation_task/data/regulatory_docs"
output_directory = "./regulation_task/data/parsed_regulatory_texts"

processor = DocProcessor(pdf_directory, output_directory)
processor.parse_directory(page_batch=5)

