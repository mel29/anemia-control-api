# Python 3.10 Slim Buster Dockerfile para una aplicación FastAPI con un modelo de Hugging Face
FROM python:3.10-slim-buster

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia los archivos de requerimientos e instala las dependencias
# --no-cache-dir: limpia el cache de pip para reducir el tamaño de la imagen.
# --upgrade pip: asegura que pip esté actualizado.
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copia tu aplicación (main.py) y el modelo (model-ml.pkl) al contenedor
COPY . /app

# Exponer el puerto que Uvicorn usará. Debe coincidir con el --port en el comando de ejecución.
EXPOSE 8000

# Comando para ejecutar la aplicación cuando se inicie el contenedor
# '0.0.0.0' es necesario para que sea accesible desde fuera del contenedor
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]