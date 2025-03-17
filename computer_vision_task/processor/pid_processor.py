# convert pdf to images
import os
from pdf2image import convert_from_path


def pdf_to_img(pdf_path, output_dir, dpi=300, fmt="jpeg"):
    os.makedirs(output_dir, exist_ok=True)
    pages = convert_from_path(pdf_path, dpi=dpi, fmt=fmt)
    for i, page in enumerate(pages):
        filename = os.path.join(output_dir, f"page_{i}.{fmt}")
        page.save(filename, fmt.upper())
        print(f"Saved {filename}")
    






