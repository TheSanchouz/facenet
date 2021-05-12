import os

import cv2
import numpy as np

from age_gender_recognition import AgeGenderDetector

dir_name = os.path.abspath(os.path.dirname(__file__))
test_image_path = os.path.join(dir_name, "single_face.jpg")
test_image = cv2.imread(test_image_path, cv2.IMREAD_COLOR)


def test_exist_single_face():
    model = AgeGenderDetector(threshold=0.5)
    model.model_load()

    result = model.process_sample(test_image)
    assert len(result) == 1


def test_correct_age_single_face():
    model = AgeGenderDetector(threshold=0.5)
    model.model_load()

    result = model.process_sample(test_image)
    box = result[0]
    predicted_age = box.age
    correct_age = 25
    average_age_error = 7

    assert correct_age - average_age_error <= predicted_age <= correct_age + average_age_error


def test_correct_gender_single_face():
    model = AgeGenderDetector(threshold=0.5)
    model.model_load()

    result = model.process_sample(test_image)
    box = result[0]
    predicted_gender, _ = list(box.gender.items())[0]
    correct_gender = 'F'

    assert correct_gender == predicted_gender


def test_empty_image():
    model = AgeGenderDetector(threshold=0.75)
    model.model_load()

    empty_input = np.zeros((416, 416, 3))
    result = model.process_sample(empty_input)
    assert result == []
