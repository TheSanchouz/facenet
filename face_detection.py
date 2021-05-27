import os

from openvino.inference_engine import IECore

from utils.utils import *
from utils.visualization import *


class FaceDetector:
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
        self.threshold = threshold

    def model_load(self):
        path_to_model_xml = os.path.join(self.model_path, "face-detection-0205.xml")
        path_to_model_bin = os.path.join(self.model_path, "face-detection-0205.bin")

        network = IECore().read_network(model=path_to_model_xml, weights=path_to_model_bin)
        self.network = IECore().load_network(network, "CPU")
        self.input_name = next(iter(self.network.input_info))
        self.output_name = sorted(self.network.outputs)[1]
        _, _, self.input_height, self.input_width = self.network.input_info[
            self.input_name
        ].input_data.shape

    def preprocess(self, data):
        preprocessed_data = []
        data_infos = []
        for img in data:
            img = np.array(img)[:, :, ::-1]

            scaled_img, scale = scale_img(
                img, [self.input_height, self.input_width]
            )
            padded_img, pad = pad_img(
                scaled_img, (0, 0, 0), [self.input_height, self.input_width],
            )

            padded_img = padded_img.transpose((2, 0, 1))
            padded_img = padded_img[np.newaxis].astype(np.float32)
            preprocessed_data.append(padded_img)
            data_infos.append((scale, pad))

        return [preprocessed_data, data_infos]

    def forward(self, data):
        data[0] = [
            self.network.infer(inputs={self.input_name: sample}) for sample in data[0]
        ]
        return data

    def postprocess(self, predictions):
        if not len(predictions[0]):
            return [[]]

        postprocessed_result = []
        for result, input_info in zip(predictions[0], predictions[1]):
            scale, pads = input_info
            h, w = self.input_height, self.input_width
            original_h = int((h - (pads[0] + pads[2])) / scale)
            original_w = int((w - (pads[1] + pads[3])) / scale)
            # print(h, w)
            # print(input_info)
            boxes = np.squeeze(result[self.output_name])

            image_predictions = []
            for box in boxes:
                if box[4] > self.threshold:
                    # print(box[0], box[0] * original_w / w)
                    image_predictions.append(
                        BoundingBox(
                            x1=int(
                                np.clip((box[0] * original_w) / w - pads[1], 0, original_w),
                            ),
                            y1=int(
                                np.clip((box[1] * original_h) / h - pads[0], 0, original_h),
                            ),
                            x2=int(
                                np.clip((box[2] * original_w) / w - pads[1], 0, original_w),
                            ),
                            y2=int(
                                np.clip((box[3] * original_h) / h - pads[0], 0, original_h),
                            ),
                            score=float(box[4]),
                        ),
                    )
            postprocessed_result.append(image_predictions)

        return postprocessed_result

    def process_sample(self, image):
        data = self.preprocess([image])
        output = self.forward(data)
        results = self.postprocess(output)
        return results[0]


if __name__ == '__main__':
    facenet = FaceDetector()
    facenet.model_load()

    # image = cv2.imread(r'd:\facenet\openvino_emotion_recognition.png', cv2.IMREAD_COLOR)
    # image = cv2.imread(r'd:\facenet\avatar.jpg', cv2.IMREAD_COLOR)
    # image = cv2.imread(r'D:\learn_materials\382006-3m\face_detection\facenet-main\816_large.jpg', cv2.IMREAD_COLOR)
    image = cv2.imread(r'D:\learn_materials\382006-3m\face_detection\facenet-main\openvino_emotion_recognition.png',
                       cv2.IMREAD_COLOR)
    boxes = facenet.process_sample(image)

    image_prop = draw_face_boxes_on_image(image, boxes)
    cv2.imshow('1', image_prop)
    cv2.waitKey()

    # cap = cv2.VideoCapture(0)
    # while True:
    #     # Capture frame-by-frame
    #     ret, frame = cap.read()
    #
    #     boxes = facenet.process_sample(frame)
    #     image = draw_face_boxes_on_image(frame, boxes)
    #     # Display the resulting frame
    #     cv2.imshow('frame', image)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
