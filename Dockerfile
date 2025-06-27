FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directory
RUN mkdir -p data/chroma_db

# Expose ports
EXPOSE 8501 8000

# Default command (can be overridden)
CMD ["sh", "-c", "streamlit run app.py --server.port 8501 --server.address 0.0.0.0 & uvicorn backend.main:app --host 0.0.0.0 --port 8000"] 