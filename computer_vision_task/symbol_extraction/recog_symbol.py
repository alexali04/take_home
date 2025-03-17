import os
import cv2
import pytesseract
import numpy as np
from cv_utils.img_utils import (
    display_img_ttb, 
    associate_text_with_symbol,
    load_img_model,
    nms,
    clamp_range,
    get_center,
    display_graph
)
import networkx as nx
from networkx.readwrite import json_graph


def detect_symbols(model, image_path, show_img: bool = False):
    """
    Detects symbols in an image using Yolov5 model.
    P&ID images are processed in tiles to disentangle close elements. 
    Each tile is processed with YOLO and then mapped back to original image space. 
    """
    image = cv2.imread(image_path)
    h_img, w_img = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((2,2), np.uint8)
    dilated = cv2.dilate(gray, kernel, iterations=1)
    processed_input = cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR)

    tile_size = 1024  # width/height of each tile
    overlap = 200     # overlap in pixels between tiles

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

            # tile-space -> global-space
            for *box, conf, cls in detections:
                xmin, ymin, xmax, ymax = map(int, box)

                global_xmin = col_start + xmin
                global_ymin = row_start + ymin
                global_xmax = col_start + xmax
                global_ymax = row_start + ymax

                all_detections.append((global_xmin, global_ymin, global_xmax, global_ymax, float(conf), int(cls)))

    final_detections = nms(all_detections, iou_threshold=0.5)

    if show_img:
        output_img = processed_input.copy()

        for (xmin, ymin, xmax, ymax, conf, cls) in final_detections:
            cv2.rectangle(output_img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
        label = f"Cls={cls}, Conf={conf:.2f}"
        cv2.putText(output_img, label, (xmin, max(ymin - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

        cv2.imshow("Tiled YOLO + Thickening", output_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return final_detections


def detect_text(image_path):
    """
    Detects text in an image using Tesseract OCR.
    """
    image = cv2.imread(image_path)

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
    
    return text_bboxes


def detect_arrows_with_strips(
    image_path, 
    strip_size=300, 
    min_length=50, 
    show_img=True
):
    """
    Detect axis-aligned arrows/lines by breaking the image into horizontal and vertical strips.
    Returns a list of arrow bounding boxes: [{'bbox': (xmin, ymin, xmax, ymax), 'orientation': 'horizontal'}, ...]

    args:
        image_path: Path to the P&ID image
        strip_size: The size (in pixels) of each strip (height for horizontal strips or width for vertical strips)
        min_length: Minimum bounding box dimension to accept as a line
        show_img: Whether to display the final overlay
    """
    original = cv2.imread(image_path)
    if original is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    h, w = original.shape[:2]

    # 1) Convert to grayscale and invert-threshold so lines become white (255) on black (0)
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    # If lines are black on white, invert them:
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    horizontal_bboxes = []
    vertical_bboxes   = []

    # A) DETECT HORIZONTAL LINES BY BREAKING INTO HORIZONTAL STRIPS
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))  # wide horizontally

    row_start = 0
    while row_start < h:
        row_end = min(row_start + strip_size, h)

        # Extract the horizontal strip from [row_start:row_end]
        strip_roi = binary[row_start:row_end, 0:w]  # full width

        # Morphological operations to highlight horizontal lines
        # Erode, then dilate
        h_lines = cv2.erode(strip_roi, h_kernel, iterations=1)
        h_lines = cv2.dilate(h_lines, h_kernel, iterations=1)

        contours, _ = cv2.findContours(h_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            x, y, w_box, h_box = cv2.boundingRect(c)

            global_x = x
            global_y = row_start + y

            if w_box < min_length and h_box < min_length:
                continue

            orientation = "horizontal"
            horizontal_bboxes.append({
                'bbox': (global_x, global_y, global_x + w_box, global_y + h_box),
                'orientation': orientation
            })

        row_start += strip_size

    # B) DETECT VERTICAL LINES BY BREAKING INTO VERTICAL STRIPS
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 30))  # tall vertically

    col_start = 0
    while col_start < w:
        col_end = min(col_start + strip_size, w)

        # Extract the vertical strip from [col_start:col_end]
        strip_roi = binary[0:h, col_start:col_end]  # full height

        # Morphological operations to highlight vertical lines
        v_lines = cv2.erode(strip_roi, v_kernel, iterations=1)
        v_lines = cv2.dilate(v_lines, v_kernel, iterations=1)

        contours, _ = cv2.findContours(v_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            x, y, w_box, h_box = cv2.boundingRect(c)
            # x,y are local to the strip, map x to global coordinate
            global_x = col_start + x
            global_y = y

            if w_box < min_length and h_box < min_length:
                continue

            orientation = "vertical"
            vertical_bboxes.append({
                'bbox': (global_x, global_y, global_x + w_box, global_y + h_box),
                'orientation': orientation
            })

        # Next vertical strip
        col_start += strip_size


    all_arrows = horizontal_bboxes + vertical_bboxes

    if show_img:
        overlay = original.copy()
        for i, arrow in enumerate(all_arrows):
            (xmin, ymin, xmax, ymax) = arrow['bbox']
            orientation = arrow['orientation']
            # color red for bounding box
            color = (0, 0, 255)
            cv2.rectangle(overlay, (xmin, ymin), (xmax, ymax), color, 2)
            label = f"A{i}_{orientation}"
            cv2.putText(
                overlay, label, 
                (xmin, max(ymin-5, 0)), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, color, 1
            )

        cv2.imshow("Arrows via horizontal/vertical strips", overlay)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return all_arrows


def build_pid_graph(symbol_bboxes, arrow_bboxes):
    """
    Build a directed graph of P&ID components and arrows.

    symbol_bboxes: list of (xmin, ymin, xmax, ymax, conf, cls, label_str)
    arrow_bboxes: list of {'bbox': (xmin, ymin, xmax, ymax), 'orientation': 'horizontal'}

    Returns a NetworkX DiGraph.
    """

    G = nx.DiGraph()

    # 1) Add nodes for each symbol
    for i, sym in enumerate(symbol_bboxes):
        xmin, ymin, xmax, ymax, conf, cls, label_str = sym
        node_id = f"symbol_{i}"

        # Store bounding box, label, class, etc. as node attributes
        G.add_node(node_id, 
                   bbox=(xmin, ymin, xmax, ymax), 
                   conf=conf, 
                   cls=cls,
                   label=label_str)

    # 2) For each arrow, connect nearest symbols
    for j, arrow in enumerate(arrow_bboxes):
        (axmin, aymin, axmax, aymax) = arrow['bbox']
        orientation = arrow['orientation']

        # Get endpoints
        if orientation == "horizontal":
            # Left endpoint and right endpoint
            centerY = (aymin + aymax) // 2
            left_pt  = (axmin, centerY)
            right_pt = (axmax, centerY)
            endpoints = [left_pt, right_pt]
        else:
            # orientation == "vertical"
            centerX = (axmin + axmax) // 2
            top_pt    = (centerX, aymin)
            bottom_pt = (centerX, aymax)
            endpoints = [top_pt, bottom_pt]

        # Find nearest symbol for each endpoint
        connected_symbols = []
        for pt in endpoints:
            min_dist = float('inf')
            nearest_symbol_id = None
            px, py = pt

            for i, sym in enumerate(symbol_bboxes):
                sxmin, symin, sxmax, symax, _, _, _ = sym
                # symbol center
                sx, sy = get_center([sxmin, symin, sxmax, symax])
                dist = np.sqrt((px - sx)**2 + (py - sy)**2)
                if dist < min_dist:
                    min_dist = dist
                    nearest_symbol_id = f"symbol_{i}"

            if nearest_symbol_id is not None:
                connected_symbols.append(nearest_symbol_id)

        # If both endpoints found a symbol, connect them
        # This can be one edge or two edges depending on your direction logic
        if len(connected_symbols) == 2:
            # For a purely undirected line, you can add one edge
            # G.add_edge(connected_symbols[0], connected_symbols[1], arrow_id=f"arrow_{j}", orientation=orientation)

            # If you want a directional edge (assuming left->right or top->bottom):
            # For horizontal: left->right
            if orientation == 'horizontal':
                # Compare axmin vs axmax
                left_sym, right_sym = connected_symbols
                # Dist check: to see which symbol is actually on the left or the right
                # or you can rely on index in endpoints
                G.add_edge(left_sym, right_sym, arrow_id=f"arrow_{j}", orientation='left-to-right')
            else:
                # vertical: top->bottom
                top_sym, bottom_sym = connected_symbols
                G.add_edge(top_sym, bottom_sym, arrow_id=f"arrow_{j}", orientation='top-to-bottom')

    return G


def construct_graphs_for_dir(image_dir, show_img: bool = False, weight_path: str = "./computer_vision_task/symbol_extraction/best.pt"):
    images = [f for f in os.listdir(image_dir) if f.lower().endswith('.jpeg') or f.lower().endswith('.jpg')]
    model = load_img_model(weight_path)

    graphs = []
    for i, image_name in enumerate(images):
        image_path = os.path.join(image_dir, image_name)

        # list of (xmin, ymin, xmax, ymax, conf, cls)
        symbols_bboxes = detect_symbols(model, image_path, show_img=False)
        print(f"Finishing detecting symbols for {image_name}")

        # list of (xmin, ymin, xmax, ymax, text)
        text_bboxes = detect_text(image_path)
        print(f"Finishing detecting text for {image_name}")

        # list of {text, text_box, symbol_box}
        associations = associate_text_with_symbol(text_bboxes, symbols_bboxes)
        print(f"Finishing associating text with symbols for {image_name}")

        # list of (xmin, ymin, xmax, ymax, conf, cls, label_str)
        enhanced_symbol_bboxes = []
        for sym in symbols_bboxes:

            (xmin, ymin, xmax, ymax, conf, cls) = sym
            # gather all text pieces that associate to this bounding box
            matched_texts = []
            for assoc in associations:
                sb = assoc['symbol_box']  # (xmin, ymin, xmax, ymax)
                if (sb[0] == xmin and sb[1] == ymin and 
                    sb[2] == xmax and sb[3] == ymax):
                    matched_texts.append(assoc['text'])
            
            label_str = " ".join(matched_texts) if matched_texts else ""
            enhanced_symbol_bboxes.append((xmin, ymin, xmax, ymax, conf, cls, label_str))

        arrow_bboxes = detect_arrows_with_strips(image_path, show_img=False)
        print(f"Finishing detecting arrows for {image_name}")

        # 5) Build the P&ID graph
        pid_graph = build_pid_graph(enhanced_symbol_bboxes, arrow_bboxes)
        breakpoint()
        print(f"Built graph for {image_name} with {len(pid_graph.nodes)} nodes and {len(pid_graph.edges)} edges.")

        graphs.append(json_graph.node_link_data(pid_graph))

        # 6) Visualize detected components
        if show_img:
            # We pass in our new 'enhanced_symbol_bboxes' plus the arrow boxes to the display function
            display_img_ttb(image_path, symbols_bboxes, text_bboxes, associations, arrow_bboxes)

            # but display reconstructed graph too
            display_graph(pid_graph, path=f"{image_dir}/graphs", name=str(i))
        
    return graphs

         


