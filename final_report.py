from ultralytics import YOLO
import json
import torch
import ultralytics.nn.tasks

torch.serialization.add_safe_globals([ultralytics.nn.tasks.DetectionModel])

model = YOLO("/content/Real-time-object-detection-with-YOLO/runs/detect/yolov8_exp1/weights/best.pt", task="detect")

results = model.val(
    data="/content/Real-time-object-detection-with-YOLO/data.yaml",  # update path if needed
    split="test",
    save_json=True,
    project="/content/Real-time-object-detection-with-YOLO/runs/detect",  # force save path
    name="val_results"  
)

metrics = {
    "mAP@0.5": results.box.map50,
    "mAP@0.5:0.95": results.box.map,
    "Precision": results.box.p.mean(),
    "Recall": results.box.r.mean(),
    "F1-Score": results.box.f1.mean()
}

with open("/content/Real-time-object-detection-with-YOLO/runs/detect/val_results/final_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("\n Final Model Evaluation Metrics:")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")
