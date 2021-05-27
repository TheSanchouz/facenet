FROM python:3.7

WORKDIR /face_detection_app

COPY . /face_detection_app

RUN pip install -r requirements-dev.txt

CMD ["python", "streamlit_app.py", "runserver", "0.0.0.0:8000"]