FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8001

WORKDIR /app

COPY asl_service/requirements.txt /app/asl_service/requirements.txt
RUN pip install --no-cache-dir -r /app/asl_service/requirements.txt

COPY asl_service/ /app/asl_service/
COPY third_party/WLASL/code/I3D/pytorch_i3d.py /app/third_party/WLASL/code/I3D/pytorch_i3d.py
COPY third_party/WLASL/code/I3D/preprocess/wlasl_class_list.txt /app/third_party/WLASL/code/I3D/preprocess/wlasl_class_list.txt

EXPOSE 8001

CMD ["sh", "-c", "python -m uvicorn asl_service.app:app --host 0.0.0.0 --port ${PORT}"]
