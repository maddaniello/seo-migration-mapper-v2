# Dockerfile per URL Migration Tool
# Ottimizzato per file di grandi dimensioni

FROM python:3.11-slim

# Installa dipendenze di sistema
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Imposta la directory di lavoro
WORKDIR /app

# Copia i file requirements
COPY requirements.txt .

# Installa le dipendenze Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copia il codice dell'applicazione
COPY app.py .
COPY .streamlit/ .streamlit/

# Crea directory per file temporanei
RUN mkdir -p /tmp/uploads

# Imposta variabili d'ambiente per ottimizzazione memoria
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=UTF-8

# Espone la porta Streamlit
EXPOSE 8501

# Healthcheck
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Comando per avviare l'applicazione
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
