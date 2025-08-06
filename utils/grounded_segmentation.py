import cv2, os
import numpy as np
import supervision as sv
import json
import torch
import torchvision
from PIL import Image
from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor


seg_root = '/mnt/localssd/models/segmentation'
det_root = '/mnt/localssd/models/detection'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# GroundingDINO config and checkpoint
GROUNDING_DINO_CONFIG_PATH = os.path.join(det_root, "GroundingDINO_SwinT_OGC.py")
GROUNDING_DINO_CHECKPOINT_PATH = os.path.join(det_root, "groundingdino_swint_ogc.pth")

# Segment-Anything checkpoint
SAM_ENCODER_VERSION = "vit_h"
SAM_CHECKPOINT_PATH = os.path.join(seg_root, "sam_vit_h_4b8939.pth")


class GroundedSegmentation:
    def __init__(self):
        
        # Building GroundingDINO inference model
        self.grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)

        # Building SAM Model and SAM Predictor
        sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
        sam.to(device=DEVICE)
        self.sam_predictor = SamPredictor(sam)
    
    def get_detected_bbx(self, obj, image_path, edit_type):
        CLASSES = [obj]
        image = cv2.imread(image_path)
        
        BOX_THRESHOLD = 0.1
        TEXT_THRESHOLD = 0.5
    
        # detect objects
        detections = self.grounding_dino_model.predict_with_classes(
            image=image,
            classes=CLASSES,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD
        )
        if edit_type == 'cut_out':
            return self.cutout_from_bbx(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), detections.xyxy, detections.confidence)

        return self.mask_from_bbx(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), detections.xyxy, detections.confidence)

    def get_most_cofident_smallest_bbx(self, boxes, confidence):
        max_conf = max(confidence)

        # Filter boxes with confidence within 0.1 of max
        selected = [
            (box, conf) for box, conf in zip(boxes, confidence)
            if max_conf - conf <= 0.1
        ]

        # Select box with smallest area
        smallest_box = min(
            selected,
            key=lambda x: (x[0][2] - x[0][0]) * (x[0][3] - x[0][1])
        )[0]

        return map(int, smallest_box)

    def cutout_from_bbx(self, image, boxes, confidence):
        """
        Crop image region from the smallest box within 0.1 confidence of the most confident box.

        Args:
            image (np.ndarray): Original image.
            boxes (List[List[float]]): Bounding boxes in [x1, y1, x2, y2] format.
            confidence (List[float]): Confidence scores for each bounding box.

        Returns:
            np.ndarray: Cropped image region.
        """
        x1, y1, x2, y2 = self.get_most_cofident_smallest_bbx(boxes, confidence)
        cropped = image[y1:y2, x1:x2]
        return Image.fromarray(cropped)


    def mask_from_bbx(self, image, boxes, confidence):
        """
        Create a binary mask from the smallest bounding box.

        Args:
            image_shape (Tuple[int, int, int]): Shape of the image (H, W, C).
            boxes (List[List[float]]): Bounding boxes in [x1, y1, x2, y2] format.

        Returns:
            np.ndarray: Binary mask (uint8, 0 or 255) for the smallest box.
        """

    

        # Find the smallest bounding box by area
        #smallest_box = min(boxes, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]))
        x1, y1, x2, y2 = self.get_most_cofident_smallest_bbx(boxes, confidence)

        # mask = np.zeros((h, w), dtype=np.uint8)
        image[y1:y2, x1:x2] = 0
        return Image.fromarray(image)

    def get_mask(self, obj, img_path):

        CLASSES = [obj]
        image = cv2.imread(img_path)
        
        BOX_THRESHOLD = 0.25
        TEXT_THRESHOLD = 0.5
        NMS_THRESHOLD = 0.8
    
        # detect objects
        detections = self.grounding_dino_model.predict_with_classes(
            image=image,
            classes=CLASSES,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD
        )
        # annotate image with detections
        box_annotator = sv.BoxAnnotator()
        for idx in range(detections.class_id.shape[0]):
            detections.class_id[idx] = 2
        
        annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections)#, labels=labels)

        #print(f"Before NMS: {len(detections.xyxy)} boxes")
        nms_idx = torchvision.ops.nms(
            torch.from_numpy(detections.xyxy), 
            torch.from_numpy(detections.confidence), 
            NMS_THRESHOLD
        ).numpy().tolist()
        
        if len(detections.xyxy) > 0:
            nms_idx = [0]
            detections.xyxy = detections.xyxy[nms_idx]
            detections.confidence = detections.confidence[nms_idx]
            detections.class_id = detections.class_id[nms_idx]

        print(f"After NMS: {len(detections.xyxy)} boxes")


        def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
            sam_predictor.set_image(image)

            if len(xyxy) == 0:
                # Return a black mask with the same height and width as the image
                height, width = image.shape[:2]
                return np.zeros((1, height, width), dtype=bool)  # or dtype=np.uint8 depending on expected output

            result_masks = []
            for box in xyxy:
                masks, scores, logits = sam_predictor.predict(
                    box=box,
                    multimask_output=True
                )
                index = np.argmax(scores)
                result_masks.append(masks[index])
            return np.array(result_masks)


        # convert detections to masks
        detections.mask = segment(
            sam_predictor=self.sam_predictor,
            image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            xyxy=detections.xyxy
        )
        return image, detections.mask[0]
    
    def cut_out_object(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Returns the cut out object from the image using the mask.
        Background is set to black (or transparent if needed).
        """
        mask = mask.astype(np.uint8)
        if mask.ndim == 2:
            mask = mask[..., np.newaxis]
        return image * mask  # keep object, remove background


    def mask_out_object(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Returns the image with the object masked out (i.e., removed).
        """
        mask_inv = (~mask).astype(np.uint8)
        if mask_inv.ndim == 2:
            mask_inv = mask_inv[..., np.newaxis]
        return image * mask_inv 
    

# gs = GroundedSegmentation()
# gs.get_mask(obj=['dog', 'cat'], img_path=["000110/000110_keyframe_0-28-33-753.jpg", "000110/000110_keyframe_0-28-40-719.jpg"])