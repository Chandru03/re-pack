# Dockerfile
FROM python:3.10-slim

# Install system deps Streamlit/Kaleido need
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libgtk2.0-dev \
        libsm6 \
        libxext6 \
        libxrender1 \
        curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project
COPY multipack_poc ./multipack_poc

# Install Python deps
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r multipack_poc/requirements.txt

# Streamlit config for container
ENV STREAMLIT_SERVER_PORT=${PORT}
ENV STREAMLIT_SERVER_ENABLECORS=false
ENV STREAMLIT_SERVER_ENABLEXSRSFPROTECTION=false
ENV STREAMLIT_SERVER_HEADLESS=true

EXPOSE 8000

CMD ["streamlit", "run", "multipack_poc/main.py", "--server.address=0.0.0.0", "--server.port=8000"]