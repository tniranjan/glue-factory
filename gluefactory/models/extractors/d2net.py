import numpy as np
import torch
import imageio.v2 as imageio
import cv2
import sys

# Importing D2Net related modules from third-party directory
sys.path.append("/home/niranjan/glue-factory/d2net/d2-net")
from lib.model_test import D2Net
from lib.utils import preprocess_image
from lib.pyramid import process_multiscale

from ..base_model import BaseModel
class D2NetExtractor(BaseModel):
    default_conf = {
        "max_kp" : 4000,
        "detection_threshold" : 0.0,
        "model_pth": "/home/niranjan/recons3d/third_party/d2net/models/d2_tf_no_phototourism.pth"
    }
    def __init__(self, conf):
        self.max_kp = conf["max_kp"]
        self.detection_threshold = conf["detection_threshold"]
        self.model = D2Net(
            model_file=conf["model_pth"],
            use_relu=True,
            use_cuda=True
        )

    def sort_and_filter(self, A, B, scores, max_rows, score_threshold):
        # Convert to numpy arrays if they aren't already
        A = np.array(A)
        B = np.array(B)
        scores = np.array(scores)

        # Sort by scores in descending order
        sorted_indices = np.argsort(scores)[::-1]
        A = A[sorted_indices]
        B = B[sorted_indices]
        scores = scores[sorted_indices]

        # Filter based on score threshold
        valid_indices = scores >= score_threshold
        A = A[valid_indices]
        B = B[valid_indices]
        scores = scores[valid_indices]

        # Limit the number of rows to max_rows
        if len(scores) > max_rows:
            A = A[:max_rows]
            B = B[:max_rows]
            scores = scores[:max_rows]

        return A, B, scores

    def _forward(self, data):
        image = data["image"]
        # for i in range(image.shape[0]):
        #     img_i = image[i,:]
            
        #     if len(img_i.shape[0]) == 1:
        #         img_i = img_i.repeat(3,1,1)
        input_image = preprocess_image(image, preprocessing="caffe")
        keypoints, scores, descriptors = process_multiscale(input_image.to(image), self.model,  scales=[1])

        return {"keypoints": keypoints, "descriptors": descriptors, "keypoint_scores": scores}


    def loss(self, pred, data):
        raise NotImplementedError