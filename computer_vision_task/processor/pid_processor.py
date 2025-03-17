import cv2
import numpy as np
import pytesseract
import networkx as nx
import os

class PIDAnalyzer:
    def __init__(self, 
                 arrowhead_template_path=None, 
                 text_detection_threshold=50,
                 arrow_detection_threshold=0.8):
        """
        Args:
            arrowhead_template_path: path to an arrowhead image for template matching
            text_detection_threshold: max distance to link text to a component
            arrow_detection_threshold: threshold for arrow template matching
        """
        self.arrowhead_template_path = arrowhead_template_path
        self.text_detection_threshold = text_detection_threshold
        self.arrow_detection_threshold = arrow_detection_threshold
    
    def preprocess_image_for_text(self, image_path):
        """Preprocess the image to improve OCR accuracy."""
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # Adaptive thresholding
        th = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
        # Optional morphological operations
        kernel = np.ones((1, 1), np.uint8)
        cleaned = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
        return cleaned

    def detect_text_regions(self, image):
        """
        Return bounding boxes for likely text regions via connected components.
        """
        num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(
            image, connectivity=8
        )
        text_bboxes = []
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            # Example heuristics - tune as needed
            if 10 < w < 3000 and 10 < h < 3000:
                text_bboxes.append((x, y, w, h))
        return text_bboxes
 
    def ocr_text_in_bboxes(self, image_path, bboxes):
        """
        Run OCR on each bounding box (ROI) and return recognized text.
        """
        image = cv2.imread(image_path)
        ocr_results = []
        for (x, y, w, h) in bboxes:
            roi = image[y:y+h, x:x+w]
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            text = pytesseract.image_to_string(roi_rgb).strip()
            if text:
                ocr_results.append({"bbox": (x, y, w, h), "text": text})
        return ocr_results

    def extract_text_data(self, image_path):
        """
        Convenience method: preprocess + detect text regions + OCR.
        Returns a list of {bbox, text}.
        """
        preprocessed = self.preprocess_image_for_text(image_path)
        text_bboxes = self.detect_text_regions(preprocessed)
        recognized_text = self.ocr_text_in_bboxes(image_path, text_bboxes)
        return recognized_text
    
    def detect_arrow_heads(self, image_path):
        """
        Detect arrowheads - returns list of (end_point_1_x, end_point_1_y, end_point_2_x, end_point_2_y)
        """
        if not self.arrowhead_template_path:
            # No template provided, skip detection or implement other logic
            return []
        
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        template = cv2.imread(self.arrowhead_template_path, cv2.IMREAD_GRAYSCALE)
        
        result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
        loc = np.where(result >= self.arrow_detection_threshold)
        template_h, template_w = template.shape[:2]
        
        arrowheads = []
        for pt in zip(*loc[::-1]):
            x, y = pt[0], pt[1]
            arrowheads.append((x, y, template_w, template_h))
        return arrowheads
    
    def detect_shapes(self, image_path):
        """
        Detect triangles, rectangles, squares, and circles in a binary or well-thresholded image.
        Returns a list of dicts like:
            [ 
            { 'shape': 'triangle', 'contour': ..., 'bbox': (x, y, w, h) }, 
            { 'shape': 'circle', ... },
            ...
            ]
        """
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
        
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected_shapes = []
        
        for cnt in contours:
            perimeter = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.001 * perimeter, True)
            vertices = len(approx)
            
            x, y, w, h = cv2.boundingRect(approx)
            
            shape_type = None
            
            if vertices == 3:
                shape_type = "triangle"
            
            elif vertices == 4:
                # Check aspect ratio to differentiate square vs rectangle
                aspect_ratio = float(w) / float(h)
                if 0.95 <= aspect_ratio <= 1.05:
                    shape_type = "square"
                else:
                    shape_type = "rectangle"

            if shape_type is not None:
                detected_shapes.append({
                    "shape": shape_type,
                    "contour": cnt,
                    "bbox": (x, y, w, h)
                })
            
            # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # cv2.putText(image, shape_type, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
            #     0.5, (0, 255, 0), 2, cv2.LINE_AA)
            # cv2.imshow("Detected Shapes", image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
        
        return detected_shapes

    def center(self, bbox):
        """
        bbox = (x, y, w, h) or (x1, y1, x2, y2) if it's a line
        This function assumes a bounding box: (x, y, w, h).
        If you have line endpoints, you might compute midpoint differently:
           midpoint_x = (x1 + x2)/2
           midpoint_y = (y1 + y2)/2
        """
        if len(bbox) == 4:    # Interpreted as x, y, w, h for a bounding box
            (x, y, w, h) = bbox
            return (x + w/2.0, y + h/2.0)
        else:                 # If you store lines as (x1, y1, x2, y2), adapt as needed
            (x1, y1, x2, y2) = bbox
            return ((x1 + x2)/2.0, (y1 + y2)/2.0)
    
    def build_pid_graph(self, text_data, arrows, components):
        """
        Create a directed graph (nx.DiGraph) where each node has:
        - text: merged natural-language description/label for that component
        - bbox: bounding box
        - component_type: e.g., "component", "arrowhead", or "unlinked_text"
        
        text_data   : list of { 'bbox': (x, y, w, h), 'text': '...' }
        arrowheads  : list of bounding boxes for arrowheads
        components  : list of { 'bbox': (x, y, w, h), 'label': 'Valve V-101', ... }
        
        The logic:
        1) For each component, find any nearby text within self.text_detection_threshold.
        2) arrows become edges
        """
        G = nx.DiGraph()

        # 1. link text elements to components
        component_nodes = {}
        for i, comp in enumerate(components):
            node_id = f"component_{i}"
            G.add_node(
                node_id,
                bbox=comp["bbox"],
                text="",  # We'll fill in with matched text below
                shape=comp["label"],  # e.g. "triangle", "circle", "rectangle"
                component_type="component"
            )
            component_nodes[node_id] = comp
        
        for t in text_data:
            t_center = self.center(t["bbox"])
            best_comp_id = None
            best_dist = float("inf")

            for node_id, comp in component_nodes.items():
                c_center = self.center(comp["bbox"])
                dist = np.linalg.norm(np.array(t_center) - np.array(c_center))
                if dist < best_dist:
                    best_dist = dist
                    best_comp_id = node_id
            
            # If it's close enough, merge the text into that component node
            if best_comp_id and best_dist < self.text_detection_threshold:
                old_text = G.nodes[best_comp_id]["text"]
                new_text = (old_text + " " + t["text"]).strip() if old_text else t["text"]
                G.nodes[best_comp_id]["text"] = new_text
            else:
                # Optional: If text doesn't match any component, you could add it as a "standalone text" node
                # For example:
                unlinked_id = f"unlinked_text_{t_center}"
                G.add_node(
                    unlinked_id,
                    bbox=t["bbox"],
                    text=t["text"],
                    shape="text_only",
                    component_type="unlinked_text"
                )

        for arrow_line in arrows:
            (x1, y1, x2, y2) = arrow_line  # e.g., line endpoints
            start_pt = (x1, y1)
            end_pt   = (x2, y2)

            # Find the nearest component to start_pt
            best_start_comp = None
            best_dist_start = float("inf")
            for node_id, comp in component_nodes.items():
                c_center = self.center(comp["bbox"])
                dist_start = np.linalg.norm(np.array(start_pt) - np.array(c_center))
                if dist_start < best_dist_start:
                    best_dist_start = dist_start
                    best_start_comp = node_id

            # Find the nearest component to end_pt
            best_end_comp = None
            best_dist_end = float("inf")
            for node_id, comp in component_nodes.items():
                c_center = self.center(comp["bbox"])
                dist_end = np.linalg.norm(np.array(end_pt) - np.array(c_center))
                if dist_end < best_dist_end:
                    best_dist_end = dist_end
                    best_end_comp = node_id

            # If both ends matched a component, create an undirected edge
            if best_start_comp and best_end_comp and best_start_comp != best_end_comp:
                # You can store attributes about the arrow if desired
                G.add_edge(best_start_comp, best_end_comp, arrow=True)

        return G

def analyze_pid(pid_path):
        """
        Example pipeline (no SOP checking here):
          1) detect components
          2) detect text
          3) detect arrowheads
          4) build graph
        """
        pid_processor = PIDAnalyzer()

        components = pid_processor.detect_shapes(pid_path)
        text_data = pid_processor.extract_text_data(pid_path)
        arrows = pid_processor.detect_arrow_heads(pid_path)

        G = pid_processor.build_pid_graph(text_data, arrows, components)
        return G

    
image_path = "./computer_vision_task/data/p&id/images/page_0.jpg"
img = cv2.imread(image_path)
if img is None:
    print("Error: Could not read image file.")
    exit()

print(os.listdir("."))


pid_analyzer = PIDAnalyzer()
shapes = pid_analyzer.detect_shapes(image_path)

# Draw bounding boxes and labels
image = img.copy()
for shape in shapes:
    x, y, w, h = shape["bbox"]
    shape_type = shape["shape"]
    
    # Draw bounding box
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Put label
    cv2.putText(image, shape_type, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, (0, 255, 0), 2, cv2.LINE_AA)

# Display the image
cv2.imshow("Detected Shapes", image)
cv2.waitKey(0)
cv2.destroyAllWindows()