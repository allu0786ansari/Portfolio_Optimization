FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends curl \\
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt ./
RUN pip install --upgrade pip && \\
    pip install --no-cache-dir streamlit plotly pandas numpy httpx python-dotenv
COPY data/        ./data/
COPY dashboard/   ./dashboard/
COPY backtesting/ ./backtesting/
COPY models/      ./models/
COPY .env         ./.env
COPY .streamlit/  ./.streamlit/
RUN groupadd -r appuser && useradd -r -g appuser -u 1000 appuser && \\
    chown -R appuser:appuser /app
USER appuser
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \\
    CMD curl -f http://localhost:8501/_stcore/health || exit 1
EXPOSE 8501
CMD ["streamlit", "run", "dashboard/app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]