FROM python:3.13.5-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# 1. Copy ONLY the requirements file first
COPY requirements.txt ./

# 2. Run pip install (this layer will now be cached)
RUN pip3 install -r requirements.txt

# 3. Download the NLTK resources
RUN python3 -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# 4. NOW copy the rest of your app
COPY app.py ./
# We can add this back now, it won't break the cache
# If you upload your 'data' folder, uncomment the next line
# COPY data/ ./data/

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run YOUR app.py file, with all flags
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableCORS=true", "--server.enableXsrfProtection=false"]