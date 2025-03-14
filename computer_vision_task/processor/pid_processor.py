import cv2
import numpy as np
import pytesseract
import networkx as nx

class PIDAnalyzer:
    def __init__(self, 
                 arrowhead_template_path=None, 
                 detection_model=None,
                 text_detection_threshold=50,
                 arrow_detection_threshold=0.8):
        """
        Args:
            arrowhead_template_path: path to an arrowhead image for template matching
            detection_model: pre-trained model for component detection (e.g., YOLO)
            text_detection_threshold: max distance to link text to a component
            arrow_detection_threshold: threshold for arrow template matching
        """
        self.arrowhead_template_path = arrowhead_template_path
        self.detection_model = detection_model
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
    
    def detect_components(self, image_path):
        """
        Detect P&ID components using a placeholder or ML detection model.
        Return a list of {bbox, label, confidence}.

        P&ID components - triangles, circles, rectangles
        """
        if not self.detection_model:
            # Placeholder: No real model, so nothing detected
            return []
        
        image = cv2.imread(image_path)
        # This depends on your detection model's API:
        results = self.detection_model.predict(image)  # Pseudocode
        components = []
        for r in results:
            components.append({
                "bbox": r["bbox"],
                "label": r["label"],
                "confidence": r["confidence"]
            })
        return components
    
    def build_pid_graph(self, text_data, arrowheads, components):
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
        2) Merge that text into the same node as the component (node["text"]).
        3) Arrowheads become separate nodes, each with component_type="arrowhead".
        4) Leftover text not matched to any component becomes a node with component_type="unlinked_text".
        """
        G = nx.DiGraph()
        
        def center(bbox):
            x, y, w, h = bbox
            return (x + w / 2.0, y + h / 2.0)
        
        unmatched_text = list(text_data)  # copy
        
        # 1) Add each component as its own node, merging text that is nearby
        for i, comp in enumerate(components):
            node_id = f"component_{i}"
            comp_bbox = comp["bbox"]
            
            G.add_node(
                node_id,
                bbox=comp_bbox,
                text=comp.get("label", ""),    # start with the detection label
                component_type="component"
            )
            
            comp_center = center(comp_bbox)
            
            matched_texts = []
            for td in unmatched_text:
                t_center = center(td["bbox"])
                dist = np.linalg.norm(np.array(comp_center) - np.array(t_center))
                if dist < self.text_detection_threshold:
                    matched_texts.append(td)
            
            if matched_texts:
                all_texts = [m["text"] for m in matched_texts]
                merged_text = " ".join(all_texts).strip()
                
                existing_label = G.nodes[node_id]["text"]
                if existing_label:
                    new_label = existing_label + " " + merged_text
                else:
                    new_label = merged_text
                
                G.nodes[node_id]["text"] = new_label
                
                for m in matched_texts:
                    unmatched_text.remove(m)
        
        # 2) Create arrowhead nodes (separate from components)
        for i, ah_bbox in enumerate(arrowheads):
            arrow_id = f"arrowhead_{i}"
            G.add_node(
                arrow_id,
                bbox=ah_bbox,
                text="arrowhead",   
                component_type="arrowhead"
            )
            

        
        return G

    
