import wandb
from ultralytics import YOLO

# Default hyperparameters (used if not running in a sweep)
DEFAULTS = {"batch": 16, "imgsz": 640, "lr0": 0.001, "epochs": 50}

# Initialize W&B project
wandb.init(project="yolov8-sweep-demo", config=DEFAULTS)
config = wandb.config

# Read values (fallback to defaults if missing)
batch = getattr(config, "batch", DEFAULTS["batch"])
imgsz = getattr(config, "imgsz", DEFAULTS["imgsz"])
lr0   = getattr(config, "lr0", DEFAULTS["lr0"])
epochs = getattr(config, "epochs", DEFAULTS["epochs"])

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Train YOLOv8 with sweep parameters
results = model.train(
    data="data.yaml",
    epochs=epochs,
    batch=batch,
    imgsz=imgsz,
    lr0=lr0,
    name=f"sweep_bs{batch}_lr{lr0:.4f}_img{imgsz}"
)

# Log best metrics to W&B
try:
    wandb.log({
        "final/mAP50": results.results_dict["metrics/mAP50(B)"],
        "final/mAP50-95": results.results_dict["metrics/mAP50-95(B)"]
    })
    print("✅ Final metrics logged to W&B")
except Exception as e:
    print(f"⚠ Could not log final results: {e}")
