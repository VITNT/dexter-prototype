FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860 \
    WLASL_RUNTIME_ROOT=/tmp/dexter-wlasl \
    WLASL_MAX_CONCURRENT_INFERENCES=1 \
    WLASL_INFERENCE_QUEUE_TIMEOUT_SECONDS=30 \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2

WORKDIR /app

COPY asl_service/requirements.txt /app/asl_service/requirements.txt
RUN pip install --no-cache-dir -r /app/asl_service/requirements.txt

RUN useradd --create-home --uid 1000 user
COPY --chown=user:user asl_service/ /app/asl_service/
USER user

EXPOSE 7860

CMD ["sh", "-c", "python -m uvicorn asl_service.app:app --host 0.0.0.0 --port ${PORT:-7860}"]
