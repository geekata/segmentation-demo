import cv2
import torch
import numpy as np
from segment_anything import sam_model_registry, SamPredictor

TYPE = "vit_b"
PATH = "checkpoints/sam_vit_b_01ec64.pth"

class SAMSegmenter:
    def __init__(self, model_type=TYPE, checkpoint_path=PATH):
        self.image = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.sam.to(self.device)
        self.predictor = SamPredictor(self.sam)

    def set_image(self, image_path):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.image = img
        self.predictor.set_image(img)

    def set_image_array(self, image_array):
        self.image = image_array
        self.predictor.set_image(image_array)

    def segment_with_box(self, box):
            masks, scores, logits = self.predictor.predict(box=box, multimask_output=True)
            return masks

    def segment_with_point(self, points, label=1):
        if len(points) == 1:
            multimask_output = True
        else:
            multimask_output = False
        labels = np.full(len(points), label, dtype=np.int32)
        masks, scores, logits = self.predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=multimask_output
        )
        best_idx = np.argmax(scores)
        return masks[best_idx] # return best mask
