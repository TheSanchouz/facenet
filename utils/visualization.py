import cv2
import numpy as np

from utils.bbox import *


def draw_face_boxes_on_image(
        image: np.ndarray,
        boxes: list
) -> np.ndarray:
    for box in boxes:
        label_str = "person" + ":" + str(np.round(box.score, 2))
        image = cv2.rectangle(
            image,
            (box.x1, box.y1),
            (box.x2, box.y2),
            (36, 255, 12),
            2
        )
        image = cv2.putText(image, label_str, (box.x1, box.y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    return image


def draw_face_boxes_with_age_and_gender_on_image(
        image: np.ndarray,
        boxes: list
) -> np.ndarray:
    for box in boxes:
        gender, gender_prob = list(box.gender.items())[0]
        gender_prob = np.round(gender_prob, 2)
        label_str = gender + ":" + str(gender_prob) + "|" + "age:" + str(box.age)

        image = cv2.rectangle(
            image,
            (box.box.x1, box.box.y1),
            (box.box.x2, box.box.y2),
            (36, 255, 12),
            2
        )
        image = cv2.putText(
            image, label_str, (box.box.x1, box.box.y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    return image
