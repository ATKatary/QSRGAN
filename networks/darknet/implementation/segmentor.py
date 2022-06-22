import cv2 
import numpy as np 

class Segmentor():
    """
    AF(weights_path, cfg_path, classes_path) = 

    Representation Invariant
        - True

    Representation Exposure
        - Safe
    """
    def __init__(self, weights_path: str, cfg_path: str, classes_path: str):
        ### Representation ###
        self.net = cv2.dnn.readNet(weights_path, cfg_path)
        self.classes = []
        with open(classes_path, 'r') as f:
            self.classes = f.read().splitlines()
        
        self.font = cv2.FONT_HERSHEY_PLAIN
        self.img = None
        self.fig_image = None
        self.roi_image = None
        self.coordinates = None
        self.color = (255, 0, 0) 
          
    def segment(self, img_path: str):
        """
        Segements an object and stores the coordinates and 

        Inputs
            :img_path: <str> the path to the image to be segmented
        """
        orig = cv2.imread(img_path)
        height, width, _ = orig.shape
        img = orig.copy()
        self.img = orig
        
        blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
        self.net.setInput(blob)
        output_layer_names = self.net.getUnconnectedOutLayersNames()
        layer_outputs = self.net.forward(output_layer_names)

        boxes = []
        confidences = []
        class_ids = []

        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores) 
                confidence = scores[class_id]
                if confidence > 0.2:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append((float(confidence)))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)

        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = str(self.classes[class_ids[i]])
                confidence = str(round(confidences[i], 2))
                cv2.rectangle(img, (x,y), (x + w, y + h), self.color, 1)
                # cv2.putText(img, f"{label} {confidence}", (x, y + 20), self.font, 1, (255, 255, 255), 1)

        self.fig_image = img
        self.coordinates = (x, y, w, h)
    
    
    def crop(self):
        """
        Crops the image to be only the segmented object
        """
        x, y, w, h = self.coordinates
        roi = self.img[y: y + h, x: x + w]
        self.roi_image = roi
