from ultralytics import YOLO
import os, sys

def main(model_path="yolov8n.pt", test_folder="test/images", conf=0.25, imgsz=640):
    if not os.path.exists(model_path):
        print(f"ERROR: model not found at {model_path}")
        sys.exit(1)

    if not os.path.isdir(test_folder):
        print(f"ERROR: test folder '{test_folder}' not found")
        sys.exit(1)

    imgs = [f for f in os.listdir(test_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not imgs:
        print("ERROR: no images found in 'test/images/' (add at least one .jpg/.png)")
        sys.exit(1)

    image_path = os.path.join(test_folder, imgs[0])
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)

    print(f"Running inference on: {image_path}")
    results = model(image_path, imgsz=imgsz, conf=conf, save=True)

    for r in results:
        print(f"Detections: {len(r.boxes)}")

    save_dir = getattr(results[0], "save_dir", None)
    if save_dir:
        print(f"Annotated results saved to: {save_dir}")
    else:
        print("Inference finished. Check the 'runs/detect' folder for outputs.")

if __name__ == "__main__":
    main()
