import os
from pdf2image import convert_from_path
import cv2
import pytesseract
import re
from sklearn.cluster import DBSCAN

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
            if not os.path.exists(f"{output_path}/pid_image_{i}.jpg"):
                image.save(f"{output_path}/pid_image_{i}.jpg", "JPEG")
        
        return [os.path.join(output_path, f) for f in os.listdir(output_path)]
    
    def detect_text(self, image_path):
        """
        Detect text using Tesseract OCR and cluster words spatially into square blobs.
        Uses DBSCAN to group words that are close together instead of relying on lines.
        Filters out words that contain only numbers and removes highly elongated text blobs.
        """
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)

        words = []
        positions = []
        n_boxes = len(data["text"])
        
        for i in range(n_boxes):
            text = data["text"][i].strip()
            if text and not re.match(r'^\d+$', text):  # Skip empty text and numbers only
                left, top, width, height = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
                right, bottom = left + width, top + height
                x_center, y_center = (left + right) // 2, (top + bottom) // 2

                words.append((text, left, top, right, bottom))
                positions.append([x_center, y_center])  # Use center positions for clustering

        # Apply DBSCAN clustering to group nearby words into square blobs
        if len(positions) > 0:
            clustering = DBSCAN(eps=50, min_samples=1).fit(positions)
            cluster_labels = clustering.labels_

            grouped_texts = {}
            for idx, label in enumerate(cluster_labels):
                text, left, top, right, bottom = words[idx]
                
                if label not in grouped_texts:
                    grouped_texts[label] = {
                        "text": text,
                        "bbox": [left, top, right, bottom]
                    }
                else:
                    grouped_texts[label]["text"] += " " + text
                    current_bbox = grouped_texts[label]["bbox"]
                    grouped_texts[label]["bbox"] = [
                        min(current_bbox[0], left),
                        min(current_bbox[1], top),
                        max(current_bbox[2], right),
                        max(current_bbox[3], bottom)
                    ]

            # Filter out elongated blobs based on aspect ratio
            filtered_texts = []
            for group in grouped_texts.values():
                bbox = group["bbox"]
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]

                aspect_ratio = width / height if height > 0 else float("inf")

                # Keep only square-ish blobs (aspect ratio between 1/8 and 8)
                if 0.125 <= aspect_ratio <= 8 and self.clean_ocr_text(group["text"]):
                    filtered_texts.append({
                        "text": self.clean_ocr_text(group["text"]),
                        "bbox": tuple(bbox)
                    })

            # Draw bounding boxes around the filtered text blobs
            for detected_text in filtered_texts:
                bbox = detected_text["bbox"]
                cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            
            cv2.imshow("Detected Text Blobs", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            return [group["bbox"] for group in filtered_texts], [group["text"] for group in filtered_texts]
        else:
            return [], []


    def clean_ocr_text(self, text):
        """
        Cleans OCR text output by removing excessive whitespace.
        """
        text = re.sub(r'[\d@.,":\-()/]', '', text)   # remove numbers, @, ., ,, :, -, (, ), /
        return text.strip()

    
    def detect_symbols(self, image_path: str):
        """
        Detects symbols using YoloV8 model. Returns bounding boxes
        """


def process_pid(pdf_path: str, output_path: str):
    """
    Process the PID and convert it to a list of images.
    """
    processor = PDFProcessor(pdf_path)
    image_paths = processor.convert_to_image(output_path)
    for image_path in image_paths:
        boxes, text_data = processor.detect_text(image_path)
        print(f"Image: {image_path}, Text_length: {len(text_data)}, 0 item length: {len(text_data[0])}")
        print(f"Text: {text_data}\n")

process_pid(
    pdf_path="./computer_vision_task/data/p&id/diagram.pdf",
    output_path="./computer_vision_task/data/p&id/images"
)