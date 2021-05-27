FROM python:3.7

WORKDIR /face_detection_app
RUN apt update && \ 
    apt-get install -y libglu1-mesa libxi-dev libxmu-dev libglu1-mesa-dev
COPY . /face_detection_app

RUN pip install -r requirements.txt

CMD streamlit run --server.port 8080 streamlit_app.py