import os
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer

# ==================================================
# 1. Configuration Parameters
# ==================================================

# Path to the model configuration file (YAML)
config_file_path = r"I:\deeplearning\detectron2\configs\COCO-Detection\faster_rcnn_R_50_FPN_3x.yaml"

# Path to the specific model checkpoint (.pth)
model_weights_path = r"I:\deeplearning\detectron2\output\randon\model_final.pth"

# Dataset Configuration
# FIX: Changed name to a unique identifier to avoid conflict with built-in datasets
test_dataset_name = "my_mica_test"
test_images_path = r"I:\deeplearning\process\coco_ALL_1010_Mica_LS\val2017"
test_json_annot_path = r"I:\deeplearning\process\coco_ALL_1010_Mica_LS\annotations\instances_val2017.json"

# Class names (Must match the classes used during training)
class_names = ['surface','line']

# ==================================================
# 2. Register Dataset
# ==================================================
try:
    # Register the dataset with the NEW unique name
    register_coco_instances(test_dataset_name, {}, test_json_annot_path, test_images_path)

    # Update Metadata for this specific dataset
    MetadataCatalog.get(test_dataset_name).set(
        thing_classes=class_names,
        evaluator_type='coco',
        json_file=test_json_annot_path,
        image_root=test_images_path
    )
    print(f"Dataset '{test_dataset_name}' registered successfully.")
except Exception as e:
    print(f"Warning during registration (might be already registered): {e}")

if __name__ == '__main__':
    # ==================================================
    # 3. Build Configuration & Load Model
    # ==================================================
    cfg = get_cfg()
    cfg.merge_from_file(config_file_path)

    # --- Model Settings ---
    cfg.MODEL.WEIGHTS = model_weights_path
    cfg.MODEL.DEVICE = "cuda"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2

    # --- Evaluation Settings ---
    # Use the unique dataset name here as well
    cfg.DATASETS.TEST = (test_dataset_name,)

    # --- Inference Settings ---
    # Optional: Set a score threshold for visualization
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

    # ==================================================
    # 4. Execute Evaluation
    # ==================================================

    # Create the Evaluator
    # Note: "bbox" is sufficient for Faster R-CNN
    evaluator = COCOEvaluator(test_dataset_name, ["bbox"], False, output_dir="./output")

    # Build the test data loader
    val_loader = build_detection_test_loader(cfg, test_dataset_name)

    # Perform inference and calculate metrics
    print(f"Evaluating model: {model_weights_path}")
    # We use DefaultPredictor to load weights cleanly, rather than DefaultTrainer
    predictor = DefaultPredictor(cfg)

    results = inference_on_dataset(predictor.model, val_loader, evaluator)

    # Print results
    print("\n=== Evaluation Results ===")
    print(results)
