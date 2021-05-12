import os

from openvino.inference_engine import IECore

from face_detection import FaceDetector
from utils.utils import *
from utils.visualization import *


class AgeGenderDetector:
    def __init__(
            self,
            model_path: str = "",
            threshold: float = 0.5,
    ):
        self.model_path = (
            model_path
            if model_path != ""
            else os.path.join(os.path.abspath(os.path.dirname(__file__)), 'models')
        )
        self.face_detector = FaceDetector(threshold=threshold)

    def model_load(self):
        path_to_model_xml = os.path.join(self.model_path, "age-gender-recognition-retail-0013.xml")
        path_to_model_bin = os.path.join(self.model_path, "age-gender-recognition-retail-0013.bin")

        network = IECore().read_network(model=path_to_model_xml, weights=path_to_model_bin)
        self.network = IECore().load_network(network, "CPU")
        self.input_name = next(iter(self.network.input_info))
        self.output_name = sorted(self.network.outputs)[1]
        _, _, self.input_height, self.input_width = self.network.input_info[
            self.input_name
        ].input_data.shape
        self.face_detector.model_load()

    def get_faces(self, data):
        preprocessed_data = self.face_detector.preprocess(data)
        face_output = self.face_detector.forward(preprocessed_data)
        face_boxes = self.face_detector.postprocess(face_output)
        return face_boxes

    def preprocess(self, data):
        face_boxes = self.get_faces(data)

        preprocessed_data = []
        preprocessed_boxes = []
        if face_boxes == [[]]:
            return [preprocessed_boxes, preprocessed_data]
        for i, img in enumerate(data):
            if not face_boxes[i]:
                preprocessed_data.append([])
                preprocessed_boxes.append([])
                continue
            preprocessed_img = []
            img_boxes = []
            img = np.array(img)[:, :, ::-1]

            for j, face_box in enumerate(face_boxes[i]):
                cropped_face = img[
                               int(face_box.y1): int(face_box.y2),
                               int(face_box.x1): int(face_box.x2),
                               ]

                scaled_img, scale = scale_img(
                    cropped_face, [self.input_height, self.input_width]
                )
                padded_img, pad = pad_img(
                    scaled_img, (0, 0, 0), [self.input_height, self.input_width],
                )

                padded_img = padded_img.transpose((2, 0, 1))
                padded_img = padded_img[np.newaxis].astype(np.float32)
                preprocessed_img.append(padded_img)
                img_boxes.append(face_box)
            preprocessed_data.append(preprocessed_img)
            preprocessed_boxes.append(img_boxes)

        return [preprocessed_data, preprocessed_boxes]

    def forward(self, data):
        data[0] = [
            [self.network.infer(inputs={self.input_name: face}) for face in sample]
            for sample in data[0]
        ]
        return data

    def postprocess(self, predictions):
        if not len(predictions[0]):
            return [[]]

        postprocessed_result = []
        for result, face_boxes in zip(predictions[0], predictions[1]):
            image_predictions = []
            for face_box, result_emotion_prob in zip(face_boxes, result):
                age = np.squeeze(result_emotion_prob["age_conv3"]) * 100
                gender = np.squeeze(result_emotion_prob["prob"])
                gender = (
                    {"F": gender[0]}
                    if gender[0] > gender[1]
                    else {"M": gender[1]}
                )

                image_predictions.append(
                    AgeGenderBoundingBox(
                        box=face_box,
                        age=int(age),
                        gender=gender
                    )
                )
            postprocessed_result.append(image_predictions)

        return postprocessed_result

    def process_sample(self, image):
        data = self.preprocess([image])
        output = self.forward(data)
        results = self.postprocess(output)
        return results[0]


if __name__ == '__main__':
    facenet = AgeGenderDetector()
    facenet.model_load()

    # image = cv2.imread(r'd:\facenet\openvino_emotion_recognition.png', cv2.IMREAD_COLOR)
    image = cv2.imread(r'd:\facenet\816_large.jpg', cv2.IMREAD_COLOR)
    # image = cv2.imread(r'd:\facenet\avatar.jpg', cv2.IMREAD_COLOR)
    boxes = facenet.process_sample(image)

    image_prop = draw_face_boxes_with_age_and_gender_on_image(image, boxes)
    cv2.imshow('1', image_prop)
    cv2.waitKey()

    # cap = cv2.VideoCapture(0)
    # while True:
    #     ret, frame = cap.read()
    #
    #     boxes = facenet.process_sample(frame)
    #     image = draw_face_boxes_with_age_and_gender_on_image(frame, boxes)
    #     # Display the resulting frame
    #     cv2.imshow('frame', image)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
