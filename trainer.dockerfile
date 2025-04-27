# Base image
FROM python:3.10

WORKDIR /workspace

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt 
COPY requirements_tests.txt requirements_tests.txt

RUN pip install --default-timeout=300 -r requirements.txt --no-cache-dir
RUN pip install -r requirements_tests.txt --no-cache-dir

COPY setup.py setup.py
COPY data.dvc data.dvc
COPY models.dvc models.dvc
COPY src/ src/
COPY tests/ tests/
COPY conf/ conf/
COPY app/ app/
COPY .dvc/config.yaml .dvc/config.yaml
COPY entrypoint.sh entrypoint.sh

RUN chmod +x entrypoint.sh

CMD ["sh", "entrypoint.sh"]




 