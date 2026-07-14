FROM mcr.microsoft.com/playwright/python:v1.49.0-jammy

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

COPY . /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        python3-dev \
        git \
        curl \
        wget && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

RUN playwright install chromium

EXPOSE 8501

RUN mkdir -p /root/.streamlit
COPY streamlit-config/secrets.toml /root/.streamlit/secrets.toml

ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
