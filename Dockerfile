FROM python:3.11-slim-bookworm
ENV PYTHONUNBUFFERED=1 TORCH_HOME=/opt/torch OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
RUN --mount=type=bind,source=.,target=/src \
    --mount=type=secret,id=ca_bundle \
    if [ -f /run/secrets/ca_bundle ]; then \
        export PIP_CERT=/run/secrets/ca_bundle SSL_CERT_FILE=/run/secrets/ca_bundle; \
    fi; \
    python -m pip install --no-cache-dir --no-compile torch==2.8.0 torchvision==0.23.0 \
        --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install --no-cache-dir --no-compile -r /src/requirements.txt \
    && python /src/modelo/descargar.py \
    && mkdir /app \
    && cp -r /src/inferencia.py /src/ejecutar.py /src/datos /app/
WORKDIR /app
ENTRYPOINT ["python", "ejecutar.py"]
