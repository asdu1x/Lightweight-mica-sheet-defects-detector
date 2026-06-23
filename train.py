import os
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultTrainer
from detectron2.utils.logger import setup_logger
from detectron2.data.datasets import register_coco_instances

# Setup logging
# setup_logger()


def get_train_cfg(config_file_path, train_dataset_name, test_dataset_name, num_classes, device, output_dir, train_num):
    """
    Build the configuration object for training.
    """
    cfg = get_cfg()
    # Load the base configuration file (e.g., Mask R-CNN, Faster R-CNN)
    cfg.merge_from_file(config_file_path)

    # --- Dataset Settings ---
    cfg.DATASETS.TRAIN = (train_dataset_name,)
    cfg.DATASETS.TEST = (test_dataset_name,)  # Used for validation during training

    # --- Dataloader & Solver Settings ---
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.SOLVER.IMS_PER_BATCH = 2  # Adjust based on GPU memory
    cfg.SOLVER.BASE_LR = 0.001  # Base learning rate

    # Calculate total iterations and step milestones based on epochs
    # Formula: (Total Images / Batch Size) * Target Epochs
    iters_per_epoch = int(train_num / cfg.SOLVER.IMS_PER_BATCH)
    cfg.SOLVER.MAX_ITER = 5000
    cfg.SOLVER.STEPS = (2500, 4000)

    # Regularization
    cfg.SOLVER.WEIGHT_DECAY = 0.0001
    cfg.SOLVER.GAMMA = 0.1  # Learning rate decay factor
    cfg.SOLVER.WARMUP_ITERS = 500  # Warmup iterations

    # --- Model Settings ---
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.DEVICE = device

    # --- Output Settings ---
    cfg.OUTPUT_DIR = output_dir

    return cfg


# ==================================================
# 1. Configuration Section (Modify these paths/values)
# ==================================================
config_file_path = r"I:\deeplearning\detectron2\configs\COCO-Detection\faster_rcnn_R_50_FPN_3x.yaml"
num_classes = 2
class_names = ['surface', 'line']
device = "cuda"  # Use "cpu" if no GPU is available
output_dir = r"I:\deeplearning\detectron2\output"

# --- Dataset Paths ---
train_dataset_name = "LP_train"
train_images_path = r"I:\deeplearning\process\coco_ALL_1010_Mica_LS\train2017"
train_json_annot_path = r"I:\deeplearning\process\coco_ALL_1010_Mica_LS\annotations\instances_train2017.json"

test_dataset_name = "LP_test"
test_images_path = r"I:\deeplearning\process\coco_ALL_1010_Mica_LS\val2017"
test_json_annot_path = r"I:\deeplearning\process\coco_ALL_1010_Mica_LS\annotations\instances_val2017.json"

# ==================================================
# 2. Register Datasets
# ==================================================
# Register the training dataset in COCO format
register_coco_instances(train_dataset_name, {}, train_json_annot_path, train_images_path)
MetadataCatalog.get(train_dataset_name).set(thing_classes=class_names)

# Register the test/validation dataset
register_coco_instances(test_dataset_name, {}, test_json_annot_path, test_images_path)
MetadataCatalog.get(test_dataset_name).set(thing_classes=class_names)

if __name__ == '__main__':
    # ==================================================
    # 3. Build Config & Start Training
    # ==================================================
    cfg = get_train_cfg(config_file_path, train_dataset_name, test_dataset_name,
                        num_classes, device, output_dir, train_num)

    # Create the Trainer
    trainer = DefaultTrainer(cfg)

    # Resume from last checkpoint if exists, or load pretrained weights
    # Set resume=True to continue training from the last saved model
    trainer.resume_or_load(resume=False)

    # Start the training loop
    print("Starting training...")
    trainer.train()

    print("Training completed!")
