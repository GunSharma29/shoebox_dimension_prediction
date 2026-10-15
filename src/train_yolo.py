"""
Model 2: YOLOv8 shoe detector.

Requires prepare_yolo_data.py to have been run first.

Run:
    python src/train_yolo.py

Produces (via ultralytics' own run folder):
    outputs/models/yolo_run/weights/best.pt
    outputs/models/yolo_run/*.png   (PR curve, confusion matrix, results.csv, etc. -- ultralytics writes these automatically)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

config.seed_everything()

from ultralytics import YOLO


def main():
    data_yaml = os.path.join(config.YOLO_DATASET_DIR, "data.yaml")
    if not os.path.exists(data_yaml):
        print("!! data.yaml not found. Run `python src/prepare_yolo_data.py` first.")
        return

    last_ckpt = os.path.join(config.MODELS_DIR, "yolo_run", "weights", "last.pt")
    resume = os.path.exists(last_ckpt)
    if resume:
        print(f"Found an interrupted previous run -- resuming from {last_ckpt} "
              f"instead of starting over.")
        model = YOLO(last_ckpt)
        results = model.train(resume=True)
    else:
        model = YOLO(config.YOLO_MODEL)   # downloads pretrained nano weights on first run
        results = model.train(
            data=data_yaml,
            epochs=config.YOLO_EPOCHS,
            imgsz=config.YOLO_IMG_SIZE,
            batch=config.YOLO_BATCH,
            project=config.MODELS_DIR,
            name="yolo_run",
            exist_ok=True,
            seed=config.SEED,
            deterministic=True,
            patience=10,
            save=True,       # ultralytics saves last.pt/best.pt after every epoch by default
            save_period=1,
        )

    # Validate and print the numbers you'll want on a slide tomorrow
    metrics = model.val()
    print("\n=== YOLO validation metrics ===")
    print(f"mAP50:    {metrics.box.map50:.3f}")
    print(f"mAP50-95: {metrics.box.map:.3f}")

    best_path = os.path.join(config.MODELS_DIR, "yolo_run", "weights", "best.pt")
    print(f"\nBest weights: {best_path}")
    print(f"Training curves / PR curve / confusion matrix auto-saved under "
          f"{os.path.join(config.MODELS_DIR, 'yolo_run')}")


if __name__ == "__main__":
    main()
