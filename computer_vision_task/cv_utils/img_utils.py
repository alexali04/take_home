"""
Warning: Result showing requires local GUI env. 
"""

import cv2
import numpy as np
import torch
import networkx as nx
import os

def load_img_model(weights_path):
    """
    Loads YOLOv5 model from torch hub. 
    """
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path)
    return model



def show_image(image):
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_center(box):
    xmin, ymin, xmax, ymax = box[:4]
    return ((xmin + xmax) / 2.0, (ymin + ymax) / 2.0)


def associate_text_with_symbol(text_bboxes, symbol_bboxes):
    symbol_centers = [get_center(b) for b in symbol_bboxes]
    associations = []

    for tb in text_bboxes:
        text_center = get_center(tb)
        min_dist = float('inf')
        closest_symbol_idx = None

        for i, sc in enumerate(symbol_centers):
            dist = np.sqrt((text_center[0] - sc[0])**2 + (text_center[1] - sc[1])**2)
            if dist < min_dist:
                min_dist = dist
                closest_symbol_idx = i
        
        if closest_symbol_idx is not None and min_dist < 200:
            associations.append({
                    'text': tb[4],
                    'text_box': tb[:4],
                    'symbol_box': symbol_bboxes[closest_symbol_idx][:4],
                    'distance': min_dist
                })

    return associations

        

def display_img_ttb(image_path, symbol_bboxes, text_bboxes, associations, arrow_bboxes=None):
    # don't overwrite original
    output_img = cv2.imread(image_path).copy()

    # draw symbol bboxes
    for (xmin, ymin, xmax, ymax, conf, cls) in symbol_bboxes:
        cv2.rectangle(output_img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
    
    # text bboxes + labeling
    for (xmin, ymin, xmax, ymax, txt) in text_bboxes:
        cv2.rectangle(output_img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(output_img, txt, (xmin, max(ymin - 5, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # draw lines - associate text w symbol
    for assoc in associations:
        tb = assoc['text_box']
        sb = assoc['symbol_box']
        text_center = ((tb[0] + tb[2]) // 2, (tb[1] + tb[3]) // 2)
        symbol_center = ((sb[0] + sb[2]) // 2, (sb[1] + sb[3]) // 2)
        cv2.line(output_img, text_center, symbol_center, (255, 0, 0), 2)
    
    # draw arrows
    for arrow in arrow_bboxes:
        cv2.rectangle(output_img, (arrow['bbox'][0], arrow['bbox'][1]), (arrow['bbox'][2], arrow['bbox'][3]), (0, 0, 255), 2)
    
    cv2.imshow('Detected Symbols & Text Associations', output_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    
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


def clamp_range(start, length, max_val):
        end = start + length
        if end > max_val:
            end = max_val
            start = max(end - length, 0)
        return start, end



import matplotlib.pyplot as plt
import networkx as nx

def display_graph(G, path, name):
    os.makedirs(path, exist_ok=True)
    pos = {}
    for node, data in G.nodes(data=True):
        (xmin, ymin, xmax, ymax) = data['bbox']
        cx = (xmin + xmax) / 2
        cy = (ymin + ymax) / 2
        pos[node] = (cx, -cy)

    labels = {n: G.nodes[n].get('label', n) for n in G.nodes()}

    plt.figure(figsize=(10, 6))
    nx.draw(G, pos=pos, labels=labels, with_labels=True, node_size=600, font_size=8, arrows=True)
    plt.title("P&ID Graph")
    plt.savefig(f"{path}/myplot_{name}.png")
    plt.close()  

import networkx as nx
from networkx.readwrite import json_graph
import json

def graph_to_adjacency_json(G: nx.DiGraph, filename="adjacency_graph.json"):
    """
    Create a simpler adjacency-based JSON.
    Each node is identified by its label (or node_id if no label).
    The edges store the connected node labels and orientation.
    """
    # 1) We'll build a dict: { "nodes": [...], "edges": { "NodeLabel": [ {"target": "OtherLabel", "orientation": "..."} ] } }
    
    label_map = {}
    for node_id, data in G.nodes(data=True):
        label = data.get("label", node_id)
        label_map[node_id] = label
    
    adjacency = {}
    for node_id in G.nodes():
        label = label_map[node_id]
        adjacency[label] = []  
    
    for source, target, data in G.edges(data=True):
        source_label = label_map[source]
        target_label = label_map[target]
        
        orientation = data.get("orientation", "undirected")
        arrow_id = data.get("arrow_id", None)
        
        adjacency[source_label].append({
            "target": target_label,
            "orientation": orientation,
            "arrow_id": arrow_id
        })
    
    final_data = {
        "nodes": [],
        "edges": adjacency
    }

    for node_id, data in G.nodes(data=True):
        label = label_map[node_id]
        node_info = {
            "label": label,
            "bbox": data.get("bbox", None),
            "class": data.get("cls", None),
            "confidence": data.get("conf", None),
        }
        final_data["nodes"].append(node_info)
    
    return final_data
