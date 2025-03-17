import cv2
import torch
import pytesseract
import numpy as np
from computer_vision_task.utils.img_utils import display_img_ttb, associate_text_with_symbol

model = torch.hub.load('ultralytics/yolov5', 'custom', path='./computer_vision_task/symbol_extraction/best.pt')

image_path = './computer_vision_task/data/p&id/images'
image = cv2.imread(f"{image_path} + /page_1.jpg")
h_img, w_img = image.shape[:2]
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Invert threshold to highlight black lines/symbols as white
_, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

# Morphological closing to connect nearby components
kernel = np.ones((5, 5), np.uint8)
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)


contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

region_bboxes = []
min_area = 50     # tweak as needed (ignore tiny specks)
max_area = 1e7    # tweak as needed (ignore absurdly large)
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    area = w * h
    if min_area < area < max_area:
        region_bboxes.append((x, y, w, h))

def expand_box(x, y, w, h, pad, maxW, maxH):
    new_x = max(x - pad, 0)
    new_y = max(y - pad, 0)
    new_w = min(w + 2*pad, maxW - new_x)
    new_h = min(h + 2*pad, maxH - new_y)
    return (new_x, new_y, new_w, new_h)

padding = 20
expanded_bboxes = []
for (bx, by, bw, bh) in region_bboxes:
    expanded_bboxes.append(expand_box(bx, by, bw, bh, padding, w_img, h_img))

# ----------------------------------------------------------------
# 6) RUN YOLO ON EACH CROPPED REGION & MAP BACK TO ORIGINAL SPACE
# ----------------------------------------------------------------
all_detections = []  # will store global (xmin, ymin, xmax, ymax, conf, cls)

for (ex, ey, ew, eh) in expanded_bboxes:
    # Crop the region from the original image
    cropped = image[ey:ey + eh, ex:ex + ew]

    # Detect with YOLO on this smaller image
    results = model(cropped)
    # results.xyxy[0]: [xmin, ymin, xmax, ymax, conf, class]
    detections = results.xyxy[0].cpu().numpy()

    for *box, conf, cls in detections:
        xmin, ymin, xmax, ymax = map(int, box)

        # Map coordinates from "cropped space" back to "original image" space
        global_xmin = ex + xmin
        global_ymin = ey + ymin
        global_xmax = ex + xmax
        global_ymax = ey + ymax

        all_detections.append((global_xmin, global_ymin, global_xmax, global_ymax, float(conf), int(cls)))

# ----------------------------------------------------------------
# 7) OPTIONAL: NON-MAX SUPPRESSION (Global)
#    Because multiple crops can overlap, we might get duplicate boxes.
#    We can do a NMS to merge duplicates or close overlaps:
# ----------------------------------------------------------------
# def nms(detections, iou_threshold=0.5):
#     # detections: list of (xmin, ymin, xmax, ymax, conf, cls)
#     boxes = np.array([d[:4] for d in detections], dtype=np.float32)
#     scores = np.array([d[4] for d in detections], dtype=np.float32)
#     idxs = cv2.dnn.NMSBoxes(
#         bboxes=boxes.tolist(), 
#         scores=scores.tolist(), 
#         score_threshold=0.1, 
#         nms_threshold=iou_threshold
#     )
#     # Collect the kept detections
#     keep = set(i[0] for i in idxs)  # .dnn.NMSBoxes returns [[idx], [idx], ...]
#     final = [d for i, d in enumerate(detections) if i in keep]
#     return final

# final_detections = nms(all_detections, iou_threshold=0.5)


output_img = image.copy()

for (xmin, ymin, xmax, ymax, conf, cls) in all_detections:
    cv2.rectangle(output_img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
    label = f"Class={cls}, Conf={conf:.2f}"
    cv2.putText(output_img, label, (xmin, max(ymin - 5, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

# Show the result (requires a local GUI environment)
cv2.imshow("Two-Pass YOLO (Segment + Crop)", output_img)
cv2.waitKey(0)
cv2.destroyAllWindows()











exit()

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

display_img_ttb(image, symbol_bboxes, text_bboxes, associations)


