from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
import cv2
import matplotlib.pyplot as plt

def main():
    # 1. configuration model
    cfg = get_cfg()
    # Load model configuration file
    cfg.merge_from_file(r"configs\COCO-Detection\faster_rcnn_R_50_FPN_3x.yaml")

    # Set model weight path
    cfg.MODEL.WEIGHTS = r"pretrained\R50-FPN.pkl"

    # Set up device (GPU)
    cfg.MODEL.DEVICE = "cuda"

    # Set the number of categories
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2

    # 2. Create a predictor
    predictor = DefaultPredictor(cfg)

    # 3. Perform inference
    # Input image path
    image_path = r"test1.jpg"

    # Read image
    im = cv2.imread(image_path)

    # Execute prediction
    outputs = predictor(im)

    # 4. Visualization results


    class_names = ['surface','line']
    v = Visualizer(im[:, :, ::-1],
                   metadata={'thing_classes':class_names},
                   scale=0.5,
                   instance_mode=ColorMode.IMAGE_BW)

    # Draw prediction results
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    # display image
    plt.figure(figsize=(14, 10))
    plt.imshow(v.get_image())
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
    
