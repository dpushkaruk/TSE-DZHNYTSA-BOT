FROM python:3.12-slim
RUN pip install --no-cache-dir mlflow==3.16.1 psycopg2-binary==2.9.11 boto3==1.40.61
