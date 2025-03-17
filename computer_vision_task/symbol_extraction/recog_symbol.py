import os
import cv2
import pytesseract
import numpy as np
from cv_utils.img_utils import (
    display_img_ttb, 
    associate_text_with_symbol,
    load_img_model
)

# model = torch.hub.load('ultralytics/yolov5', 'custom', path='./computer_vision_task/symbol_extraction/best.pt')

# image_path = './computer_vision_task/data/p&id/images'
# image = cv2.imread(f"{image_path} + /page_1.jpg")
# h_img, w_img = image.shape[:2]
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # Invert threshold to highlight black lines/symbols as white
# _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

# # Morphological closing to connect nearby components
# kernel = np.ones((5, 5), np.uint8)
# closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)


# contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# region_bboxes = []
# min_area = 50     # tweak as needed (ignore tiny specks)
# max_area = 1e7    # tweak as needed (ignore absurdly large)
# for c in contours:
#     x, y, w, h = cv2.boundingRect(c)
#     area = w * h
#     if min_area < area < max_area:
#         region_bboxes.append((x, y, w, h))

# def expand_box(x, y, w, h, pad, maxW, maxH):
#     new_x = max(x - pad, 0)
#     new_y = max(y - pad, 0)
#     new_w = min(w + 2*pad, maxW - new_x)
#     new_h = min(h + 2*pad, maxH - new_y)
#     return (new_x, new_y, new_w, new_h)

# padding = 20
# expanded_bboxes = []
# for (bx, by, bw, bh) in region_bboxes:
#     expanded_bboxes.append(expand_box(bx, by, bw, bh, padding, w_img, h_img))

# # ----------------------------------------------------------------
# # 6) RUN YOLO ON EACH CROPPED REGION & MAP BACK TO ORIGINAL SPACE
# # ----------------------------------------------------------------
# all_detections = []  # will store global (xmin, ymin, xmax, ymax, conf, cls)

# for (ex, ey, ew, eh) in expanded_bboxes:
#     # Crop the region from the original image
#     cropped = image[ey:ey + eh, ex:ex + ew]

#     # Detect with YOLO on this smaller image
#     results = model(cropped)
#     # results.xyxy[0]: [xmin, ymin, xmax, ymax, conf, class]
#     detections = results.xyxy[0].cpu().numpy()

#     for *box, conf, cls in detections:
#         xmin, ymin, xmax, ymax = map(int, box)

#         # Map coordinates from "cropped space" back to "original image" space
#         global_xmin = ex + xmin
#         global_ymin = ey + ymin
#         global_xmax = ex + xmax
#         global_ymax = ey + ymax

#         all_detections.append((global_xmin, global_ymin, global_xmax, global_ymax, float(conf), int(cls)))

# # ----------------------------------------------------------------
# # 7) OPTIONAL: NON-MAX SUPPRESSION (Global)
# #    Because multiple crops can overlap, we might get duplicate boxes.
# #    We can do a NMS to merge duplicates or close overlaps:
# # ----------------------------------------------------------------
# # def nms(detections, iou_threshold=0.5):
# #     # detections: list of (xmin, ymin, xmax, ymax, conf, cls)
# #     boxes = np.array([d[:4] for d in detections], dtype=np.float32)
# #     scores = np.array([d[4] for d in detections], dtype=np.float32)
# #     idxs = cv2.dnn.NMSBoxes(
# #         bboxes=boxes.tolist(), 
# #         scores=scores.tolist(), 
# #         score_threshold=0.1, 
# #         nms_threshold=iou_threshold
# #     )
# #     # Collect the kept detections
# #     keep = set(i[0] for i in idxs)  # .dnn.NMSBoxes returns [[idx], [idx], ...]
# #     final = [d for i, d in enumerate(detections) if i in keep]
# #     return final

# # final_detections = nms(all_detections, iou_threshold=0.5)


# output_img = image.copy()

# for (xmin, ymin, xmax, ymax, conf, cls) in all_detections:
#     cv2.rectangle(output_img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
#     label = f"Class={cls}, Conf={conf:.2f}"
#     cv2.putText(output_img, label, (xmin, max(ymin - 5, 0)),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

# # Show the result (requires a local GUI environment)
# cv2.imshow("Two-Pass YOLO (Segment + Crop)", output_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


def detect_symbols_and_text(model, image_path, show_img: bool = False):
    image = cv2.imread(image_path)
    h_img, w_img = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((2,2), np.uint8)
    dilated = cv2.dilate(gray, kernel, iterations=1)
    processed_input = cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR)

    tile_size = 1024  # width/height of each tile
    overlap = 200     # overlap in pixels between tiles

    # A function to ensure we don't exceed the image boundary
    def clamp_range(start, length, max_val):
        end = start + length
        if end > max_val:
            end = max_val
            start = max(end - length, 0)
        return start, end
    
    all_detections = []  # will collect (xmin, ymin, xmax, ymax, conf, cls)

    for row in range(0, h_img, tile_size - overlap):
        # clamp row to not exceed image bottom
        row_start, row_end = clamp_range(row, tile_size, h_img)
        if row_start >= h_img:
            break

        for col in range(0, w_img, tile_size - overlap):
            # clamp col to not exceed image right side
            col_start, col_end = clamp_range(col, tile_size, w_img)
            if col_start >= w_img:
                break

            # Crop the tile
            tile = processed_input[row_start:row_end, col_start:col_end]

            # Run YOLO on this tile
            results = model(tile)
            detections = results.xyxy[0].cpu().numpy()

            # Map each detection from tile-space -> global-space
            for *box, conf, cls in detections:
                xmin, ymin, xmax, ymax = map(int, box)

                # Convert local tile coords to global image coords
                global_xmin = col_start + xmin
                global_ymin = row_start + ymin
                global_xmax = col_start + xmax
                global_ymax = row_start + ymax

                all_detections.append((global_xmin, global_ymin, global_xmax, global_ymax, float(conf), int(cls)))

    # ----------------------------------------------------------------
    # 5) OPTIONAL: GLOBAL NMS
    #    We can use OpenCV's DNN NMS or a custom one to merge duplicates
    # ----------------------------------------------------------------
    def nms(detections, iou_threshold=0.5, score_threshold=0.1):
        """
        detections: list of (xmin, ymin, xmax, ymax, conf, cls)
        """
        if len(detections) == 0:
            return []

        boxes = np.array([d[:4] for d in detections], dtype=np.float32)
        scores = np.array([d[4] for d in detections], dtype=np.float32)
        
        # Convert to list of [x, y, width, height]
        # If your detection is (xmin, ymin, xmax, ymax),
        # NMSBoxes expects [x, y, width, height].
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        bboxes_for_nms = []
        for i in range(len(boxes)):
            x, y = boxes[i, 0], boxes[i, 1]
            w, h = widths[i], heights[i]
            bboxes_for_nms.append([float(x), float(y), float(w), float(h)])
        
        # Run NMS
        indices = cv2.dnn.NMSBoxes(
            bboxes=bboxes_for_nms,
            scores=scores.tolist(),
            score_threshold=score_threshold,
            nms_threshold=iou_threshold
        )
        
        # If indices is empty or None, return an empty list
        if indices is None or len(indices) == 0:
            return []
        
        # Otherwise, indices is something like [[0], [2], ...]
        # Flatten it and use it to index
        if isinstance(indices, np.ndarray):
            # In recent OpenCV versions, indices might be an array of shape (N,1)
            indices = indices.flatten()
        else:
            # If it's a list of lists
            indices = [i[0] for i in indices]

        final = [detections[i] for i in indices]
        return final

    final_detections = nms(all_detections, iou_threshold=0.5)

    output_img = processed_input.copy()

    for (xmin, ymin, xmax, ymax, conf, cls) in final_detections:
        cv2.rectangle(output_img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
        label = f"Cls={cls}, Conf={conf:.2f}"
        cv2.putText(output_img, label, (xmin, max(ymin - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

    cv2.imshow("Tiled YOLO + Thickening", output_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




    # # 1st pass: detect large symbols
    # results = model(processed_input)
    # detections = results.xyxy[0].cpu().numpy()

    # symbol_bboxes = []
    # for *box, conf, cls in detections:
    #     xmin, ymin, xmax, ymax = map(int, box)
    #     symbol_bboxes.append((xmin, ymin, xmax, ymax, conf, int(cls)))
    
    # # Display results from first pass
    # output_img = image.copy()
    
    # for (xmin, ymin, xmax, ymax, conf, cls) in symbol_bboxes:
    #     cv2.rectangle(output_img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
    #     label = f"Class={cls}, Conf={conf:.2f}"
    #     cv2.putText(output_img, label, (xmin, max(ymin - 5, 0)),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # if show_img:
    #     cv2.imshow("First Pass YOLO Detection", output_img)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    
    # return
   
   
   
   
    # # Invert threshold to highlight black lines/symbols as white
    # _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # # Morphological closing to connect nearby components
    # kernel = np.ones((5, 5), np.uint8)
    # closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # region_bboxes = []
    # min_area = 50     # tweak as needed (ignore tiny specks)
    # max_area = 1e7    # tweak as needed (ignore absurdly large)
    # for c in contours:
    #     x, y, w, h = cv2.boundingRect(c)
    #     area = w * h
    #     if min_area < area < max_area:
    #         region_bboxes.append((x, y, w, h))

    # def expand_box(x, y, w, h, pad, maxW, maxH):
    #     new_x = max(x - pad, 0)
    #     new_y = max(y - pad, 0)
    #     new_w = min(w + 2*pad, maxW - new_x)
    #     new_h = min(h + 2*pad, maxH - new_y)
    #     return (new_x, new_y, new_w, new_h)

    # padding = 20
    # expanded_bboxes = []
    # for (bx, by, bw, bh) in region_bboxes:
    #     expanded_bboxes.append(expand_box(bx, by, bw, bh, padding, w_img, h_img))

    # # ----------------------------------------------------------------
    # # 6) RUN YOLO ON EACH CROPPED REGION & MAP BACK TO ORIGINAL SPACE
    # # ----------------------------------------------------------------
    # all_detections = []  # will store global (xmin, ymin, xmax, ymax, conf, cls)

    # for (ex, ey, ew, eh) in expanded_bboxes:
    #     # Crop the region from the original image
    #     cropped = image[ey:ey + eh, ex:ex + ew]

    #     # Detect with YOLO on this smaller image
    #     results = model(cropped)
    #     # results.xyxy[0]: [xmin, ymin, xmax, ymax, conf, class]
    #     detections = results.xyxy[0].cpu().numpy()

    #     for *box, conf, cls in detections:
    #         xmin, ymin, xmax, ymax = map(int, box)

    #         # Map coordinates from "cropped space" back to "original image" space
    #         global_xmin = ex + xmin
    #         global_ymin = ey + ymin
    #         global_xmax = ex + xmax
    #         global_ymax = ey + ymax

    #         all_detections.append((global_xmin, global_ymin, global_xmax, global_ymax, float(conf), int(cls)))

    # # ----------------------------------------------------------------
    # # 7) OPTIONAL: NON-MAX SUPPRESSION (Global)
    # #    Because multiple crops can overlap, we might get duplicate boxes.
    # #    We can do a NMS to merge duplicates or close overlaps:
    # # ----------------------------------------------------------------
    # # def nms(detections, iou_threshold=0.5):
    # #     # detections: list of (xmin, ymin, xmax, ymax, conf, cls)
    # #     boxes = np.array([d[:4] for d in detections], dtype=np.float32)
    # #     scores = np.array([d[4] for d in detections], dtype=np.float32)
    # #     idxs = cv2.dnn.NMSBoxes(
    # #         bboxes=boxes.tolist(), 
    # #         scores=scores.tolist(), 
    # #         score_threshold=0.1, 
    # #         nms_threshold=iou_threshold
    # #     )
    # #     # Collect the kept detections
    # #     keep = set(i[0] for i in idxs)  # .dnn.NMSBoxes returns [[idx], [idx], ...]
    # #     final = [d for i, d in enumerate(detections) if i in keep]
    # #     return final

    # # final_detections = nms(all_detections, iou_threshold=0.5)


    # output_img = image.copy()

    # for (xmin, ymin, xmax, ymax, conf, cls) in all_detections:
    #     cv2.rectangle(output_img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
    #     label = f"Class={cls}, Conf={conf:.2f}"
    #     cv2.putText(output_img, label, (xmin, max(ymin - 5, 0)),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # # Show the result (requires a local GUI environment)
    # cv2.imshow("Two-Pass YOLO (Segment + Crop)", output_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def detect_symbols_text_for_dir(image_dir, show_img: bool = False, weight_path: str = "./computer_vision_task/symbol_extraction/best.pt"):
    images = [f for f in os.listdir(image_dir) if f.lower().endswith('.jpeg') or f.lower().endswith('.jpg')]

    model = load_img_model(weight_path)

    for image in images:
        image_path = os.path.join(image_dir, image)
        detect_symbols_and_text(model, image_path, show_img)
    









# exit()

# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)     # converts to mostly black image

# results = model(image)

# detections = results.xyxy[0].cpu().numpy()

# symbol_bboxes = []
# for *box, conf, cls in detections:
#     xmin, ymin, xmax, ymax = map(int, box)
#     symbol_bboxes.append((xmin, ymin, xmax, ymax, conf, int(cls)))


# config_tesseract = "--psm 6"
# data = pytesseract.image_to_data(image, config=config_tesseract, output_type=pytesseract.Output.DICT)
# text_bboxes = []
# num_boxes = len(data['level'])

# for i in range(num_boxes):
#     # If confidence is high enough and the text is not empty
#     if int(data['conf'][i]) > 60 and data['text'][i].strip() != '':
#         x = data['left'][i]
#         y = data['top'][i]
#         w = data['width'][i]
#         h = data['height'][i]
#         recognized_text = data['text'][i]

#         # Store (xmin, ymin, xmax, ymax, text)
#         text_bboxes.append((x, y, x + w, y + h, recognized_text))

# associations = associate_text_with_symbol(text_bboxes, symbol_bboxes)

# display_img_ttb(image, symbol_bboxes, text_bboxes, associations)


