FROM python:3.11-slim

WORKDIR /app

# Copiamos requirements y los instalamos
COPY requirements.txt .
# Dependencias de sistema para OpenCV (libGL y libglib)
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

# Copiamos el resto del proyecto
COPY . .

# Puerto expuesto (coincide con docker-compose)
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"] 