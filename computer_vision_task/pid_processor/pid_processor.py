import os
from pdf2image import convert_from_path

class PDFProcessor:
    def __init__(self, pdf_path: str):
        """
        Given Piping & Instrumentation Diagram (PID) in PDF format, process the PDF and extract the information.
        """
        self.pdf_path = pdf_path

    def convert_to_image(self, output_path: str):
        """
        Convert the PDF to a list of images.
        """
        images = convert_from_path(self.pdf_path)

        os.makedirs(output_path, exist_ok=True)

        for i, image in enumerate(images):
            image.save(f"{output_path}/pid_image_{i}.jpg", "JPEG")


def process_pid(pdf_path: str, output_path: str):
    """
    Process the PID and convert it to a list of images.
    """
    processor = PDFProcessor(pdf_path)
    processor.convert_to_image(output_path)

process_pid(
    pdf_path="./computer_vision_task/data/p&id/diagram.pdf",
    output_path="./computer_vision_task/data/p&id/images"
)