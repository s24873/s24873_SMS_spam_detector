FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update && apt-get install -y openjdk-17-jre && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

COPY implementations/ml_api.py /app/ml_api.py

COPY ./models /app/models

EXPOSE 5000

CMD ["python", "ml_api.py"]
