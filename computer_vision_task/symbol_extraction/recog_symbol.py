# load best.pt into yolov5 model


import cv2
import torch
import pytesseract
import numpy as np
from computer_vision_task.utils.img_utils import show_image, get_center, display_img_ttb, associate_text_with_symbol

model = torch.hub.load('ultralytics/yolov5', 'custom', path='./computer_vision_task/symbol_extraction/best.pt')

image_path = './computer_vision_task/data/p&id/images/page_0.jpg'
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)     # converts to mostly black image

results = model(image)

detections = results.xyxy[0].cpu().numpy()

symbol_bboxes = []
for *box, conf, cls in detections:
    xmin, ymin, xmax, ymax = map(int, box)
    symbol_bboxes.append((xmin, ymin, xmax, ymax, conf, int(cls)))


config_tesseract = "--psm 6"
data = pytesseract.image_to_data(image, config=config_tesseract, output_type=pytesseract.Output.DICT)
text_bboxes = []
num_boxes = len(data['level'])
for i in range(num_boxes):
    # If confidence is high enough and the text is not empty
    if int(data['conf'][i]) > 60 and data['text'][i].strip() != '':
        x = data['left'][i]
        y = data['top'][i]
        w = data['width'][i]
        h = data['height'][i]
        recognized_text = data['text'][i]

        # Store (xmin, ymin, xmax, ymax, text)
        text_bboxes.append((x, y, x + w, y + h, recognized_text))

associations = associate_text_with_symbol(text_bboxes, symbol_bboxes)

# Assume you have the following lists from your detection/association pipeline:
#   1) symbol_bboxes = [(xmin, ymin, xmax, ymax, conf, cls), ...]
#   2) text_bboxes   = [(xmin, ymin, xmax, ymax, text), ...]
#   3) associations  = [{'text': ..., 'text_box': ..., 'symbol_box': ..., 'distance': ...}, ...]

# Make a copy so we don't overwrite the original
display_img_ttb(image, symbol_bboxes, text_bboxes, associations)


