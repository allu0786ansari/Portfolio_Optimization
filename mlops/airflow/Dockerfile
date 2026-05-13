FROM python:3.11-slim-bullseye

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

ENV AIRFLOW_HOME=/opt/airflow
ENV PATH="/root/.local/bin:${PATH}"

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    "apache-airflow==2.9.0" \
    --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.9.0/constraints-3.11.txt"

RUN pip install --no-cache-dir \
    requests==2.31.0 \
    "mlflow-skinny==2.11.1" \
    loguru==0.7.3 \
    "pandas==2.2.3"

RUN mkdir -p /opt/airflow/dags /opt/airflow/logs /opt/airflow/plugins

WORKDIR /opt/airflow