FROM python:3.10-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 PORT=8501

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py streamlit_app.py shopping_preference_model.pkl ./

EXPOSE 8000 8501

CMD uvicorn app:app --host 0.0.0.0 --port 8000 & \
    streamlit run streamlit_app.py --server.port ${PORT} --server.address 0.0.0.0 --server.headless true