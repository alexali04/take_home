import os
import json
import fitz 
import re
from typing import Optional

from utils.prompting import Regulatory_API_Prompt, extract_clauses_from_docx


class Doc_Processor:
    def __init__(self, pdf_directory="./regulation_task/data/regulations", output_directory="./regulation_task/data/regulatory_texts"):
        """
        Initialize the PDFParser with a directory of PDFs.
        args:
           (str) pdf_directory: Path to the directory containing PDFs.
           (str) output_directory: Path to store extracted JSON files with chunked text.
        """
        self.pdf_directory = pdf_directory
        self.output_directory = output_directory
        os.makedirs(self.output_directory, exist_ok=True)
    

    def clean_text(self, text):
        """
        Cleans extracted text by removing boilerplate, fixing formatting, and removing hyphenated words.
        """
        text = re.sub(r"PO\s+\d{5}\s+Frm\s+\d+\s+Fmt\s+\d+\s+Sfmt\s+\d+", "", text)
        text = re.sub(r"EDITORIAL NOTE:.*?(www\.\S+)?", "", text, flags=re.DOTALL)
        text = re.sub(r"List of CFR Sections Affected.*?(www\.\S+)?", "", text, flags=re.DOTALL)
        text = re.sub(r"FEDERAL REGISTER citations.*?(www\.\S+)?", "", text, flags=re.DOTALL)
        text = re.sub(r"AUTHORITY:.*", "", text)
        text = re.sub(r"SOURCE:.*", "", text)
        text = re.sub(r"EFFECTIVE DATE NOTE:.*", "", text)
        text = re.sub(r"PART\s+\d+\s+\[RESERVED\]", "", text)
        text = re.sub(r"(\w+)-\s+(\w+)", r"\1\2", text)  # Fix hyphenated words
        text = re.sub(r"\n\s*\n", "\n", text).strip()
        return text
    

    def parse_directory_of_pdfs(
        self, 
        page_batch=5, 
        max_pdfs: Optional[int] = None,
        api_prompter: Regulatory_API_Prompt = None,
        use_llm: bool = False
    ):
        """
        Parse all PDFs in the directory.
        """
        pdf_files = [f for f in os.listdir(self.pdf_directory) if f.endswith(".pdf")]
        if not pdf_files:
            print("No PDF files found in the directory.")
            return
        
        if max_pdfs is not None:
            pdf_files = pdf_files[:max_pdfs]   # Only process the first max_pdfs PDFs

        for pdf in pdf_files:
            pdf_path = os.path.join(self.pdf_directory, pdf)
            self.parse_pdf(
                pdf_path, 
                page_batch=page_batch,
                api_prompter=api_prompter
            )


    def parse_pdf(
        self, 
        pdf_path, 
        page_batch=10,
        api_prompter: Regulatory_API_Prompt = None,
        use_llm: bool = False
    ):
        """
        Parse a PDF file and save the extracted, chunked text to a JSON file.
        """
        pdf_name = os.path.basename(pdf_path).replace(".pdf", "")
        output_file = os.path.join(self.output_directory, f"{pdf_name}.json")

        try:
            doc = fitz.open(pdf_path)
            total_pages = len(doc)

            text_batches = []
            for start_page in range(0, total_pages, page_batch):
                end_page = min(start_page + page_batch, total_pages)
                text_batch = [self.clean_text(doc[page].get_text("text")) for page in range(start_page, end_page)]
                text_batches.append("\n".join(text_batch))
                        
            chunked_clauses_str = self.chunk_to_regulatory_clauses(
                document="\n\n".join(text_batches)[:1000],         # remove
                use_llm=use_llm,
                api_prompter=api_prompter
            )

            clauses_dict = json.loads(chunked_clauses_str)
            

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump({"document_name": pdf_name, "clauses": clauses_dict["clauses"]}, f, indent=4) 
            
            print(f"Chunked JSON saved to: {output_file}")
            return output_file

        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")
            return None

    
    def chunk_to_regulatory_clauses(
        self, 
        document: str, 
        use_llm: bool = False,
        api_prompter: Regulatory_API_Prompt = None
    ) -> list[str]:
        """
        Chunk a regulatory document into clauses.

        WARNING: LLM chunking is EXPENSIVE! Only use if data is highly unstructured. 
        """
        if use_llm:
            return document.split("\n\n")
     
        api_prompter.reset_content()    # make sure we don't append content for every chunk
        
        return extract_clauses_from_docx(api_prompter, document)
        

def extract_regulation_clauses(args, data_dir: str):
    """
    Process all PDFs in the regulations directory and save the extracted, chunked clauses to JSON directory. 
    """

    if args.max_pdf_count == -1:
        max_pdf_count = None
    else:
        max_pdf_count = args.max_pdf_count

    reg_dir = os.path.join(data_dir, "regulations")
    json_regulatory_dir = os.path.join(data_dir, "regulatory_json")
    prompt_dir = os.path.join(data_dir, "prompts")

    chunking_model = args.chunking_model
    sys_prompt = open(f"{prompt_dir}/{args.sys_prompt}.txt").read()
    content = open(f"{prompt_dir}/{args.reg_prompt}.txt").read()


    document_processor = Doc_Processor(
        pdf_directory=reg_dir,
        output_directory=json_regulatory_dir
    )

    api_prompter = Regulatory_API_Prompt(
        model=chunking_model,
        sys_prompt=sys_prompt,
        content=content,
        max_tokens=args.max_tokens
    )

    document_processor.parse_directory_of_pdfs(
        page_batch=args.page_batch,
        max_pdfs=max_pdf_count,
        api_prompter=api_prompter,
        use_llm=args.use_llm_chunking
    )

    return json_regulatory_dir
    