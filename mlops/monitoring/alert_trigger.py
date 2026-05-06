"""Webhook server: receives Grafana alerts and triggers Airflow DAG.

Run: python mlops/monitoring/alert_trigger.py
Listens: http://localhost:5001/alert
Grafana webhook URL: http://host.docker.internal:5001/alert
"""
import os
import subprocess
from pathlib import Path

from flask import Flask, jsonify, request
from loguru import logger

app = Flask(__name__)
AIRFLOW_HOME = str(Path("mlops/airflow").resolve())
DAG_ID = "retrain_portfolio_agent"


@app.route("/alert", methods=["POST"])
def receive_alert():
    data  = request.get_json(silent=True) or {}
    state = data.get("state", "alerting")
    logger.info(f"Alert: state={state}")
    if state == "alerting":
        result = subprocess.run(
            ["airflow", "dags", "trigger", DAG_ID],
            env={**os.environ, "AIRFLOW_HOME": AIRFLOW_HOME},
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            logger.info("DAG triggered successfully")
            return jsonify({"status": "triggered"}), 200
        logger.error(f"Trigger failed: {result.stderr}")
        return jsonify({"status": "error"}), 500
    return jsonify({"status": "ignored"}), 200


@app.route("/health")
def health():
    return jsonify({"status": "ok"}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)