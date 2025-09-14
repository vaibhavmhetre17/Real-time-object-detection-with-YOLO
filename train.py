from ultralytics import YOLO

# Load a YOLOv8 model (you already have yolov8n.pt in your folder)
model = YOLO("yolov8n.pt")

# Train the model
model.train(
    data="data.yaml",   
    epochs=50,          
    batch=16,           
    imgsz=640,          
    name="yolov8_exp1"  
)
