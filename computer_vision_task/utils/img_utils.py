import cv2
import numpy as np

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
        
        if closest_symbol_idx is not None:
            associations.append({
                    'text': tb[4],
                    'text_box': tb[:4],
                    'symbol_box': symbol_bboxes[closest_symbol_idx][:4],
                    'distance': min_dist
                })

    return associations


        

def display_img_ttb(image, symbol_bboxes, text_bboxes, associations):
    # don't overwrite original
    output_img = image.copy()

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
    
    cv2.imshow('Detected Symbols & Text Associations', output_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    
    