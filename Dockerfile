# Base image
FROM python:3.10

WORKDIR /workspace

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY setup.py setup.py

RUN pip install --default-timeout=300 --no-cache-dir --upgrade -r requirements.txt 


COPY models/deployable_model.pt models/deployable_model.pt
COPY src/ src/
COPY conf/ conf/
COPY app/ app/

CMD ["uvicorn", "app.cloud_deployment:app", "--host", "0.0.0.0", "--port", "80"]
