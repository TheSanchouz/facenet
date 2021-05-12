import os

import cv2
import numpy as np
import pytest

from face_detection import FaceDetector

dir_name = os.path.abspath(os.path.dirname(__file__))
test_image_path = os.path.join(dir_name, "single_face.jpg")
test_image = cv2.imread(test_image_path, cv2.IMREAD_COLOR)


def test_exist_single_face():
    model = FaceDetector(threshold=0.5)
    model.model_load()

    result = model.process_sample(test_image)
    assert len(result) == 1


def test_empty_image():
    model = FaceDetector(threshold=0.75)
    model.model_load()

    empty_input = np.zeros((416, 416, 3))
    result = model.process_sample(empty_input)
    assert result == []
