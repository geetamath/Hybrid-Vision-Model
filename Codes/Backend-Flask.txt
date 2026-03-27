from flask import Flask, request, send_file, render_template
import cv2
import numpy as np
import os

app = Flask(__name__)

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load YOLO model
yolo_net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")
yolo_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
yolo_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Load YOLO class names
with open("coco.names", "r") as f:
    classes = f.read().strip().split("\n")

UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return "No image uploaded", 400
    
    file = request.files['image']
    filename = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filename)
    
    # Read the uploaded image
    image = cv2.imread(filename)
    if image is None:
        return "Error loading image", 400
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Face Detection
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(image, "Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    # Object Detection with YOLO
    height, width = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    yolo_net.setInput(blob)
    layer_names = yolo_net.getUnconnectedOutLayersNames()
    yolo_detections = yolo_net.forward(layer_names)

    # Process YOLO detections
    for detection in yolo_detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.3:
                center_x, center_y, box_w, box_h = (obj[0:4] * [width, height, width, height]).astype("int")
                x = int(center_x - (box_w / 2))
                y = int(center_y - (box_h / 2))
                label = classes[class_id]
                color = (0, 255, 0)
                
                # Draw YOLO detection
                cv2.rectangle(image, (x, y), (x + box_w, y + box_h), color, 2)
                cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Save output image
    output_path = os.path.join(UPLOAD_FOLDER, "output.jpg")
    cv2.imwrite(output_path, image)
    
    return send_file(output_path, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
