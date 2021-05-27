import streamlit as st
from PIL import Image
from utils.visualization import *

from age_gender_recognition import AgeGenderDetector

DEFAULT_CONFIDENCE_THRESHOLD = 0.5


if __name__ == '__main__':
    st.title("Welcome to the age gender recognition!")
    st.subheader("This model will recognize faces in the image and determine the gender and age of the person.")
    st.text('Step 1. Please upload your image.')
    uploaded_file = st.file_uploader("Upload Files", type=['png', 'jpeg', 'jpg'])
    if uploaded_file is not None:
        file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type, "FileSize": uploaded_file.size}
        st.text('Check: Image details is:')
        st.write(file_details)
    image = np.array(Image.open(uploaded_file))

    detection_confidence_threshold = st.slider(
        "Step 2. Set detection confidence threshold.", 0.0, 1.0, DEFAULT_CONFIDENCE_THRESHOLD, 0.01,
    )

    facenet = AgeGenderDetector(threshold=detection_confidence_threshold)
    facenet.model_load()

    boxes = facenet.process_sample(image)

    image_prop = draw_face_boxes_with_age_and_gender_on_image(image, boxes)
    st.text('Step 3. Keep calm and see result.')
    st.image(image_prop, channels="RGB", caption="Result of model working.")
    st.text('Please give us credit pass')
    st.write(":smile:")
