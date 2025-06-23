## ANEMIA CONTROL API

### Introducción

Este repositorio contiene el código base de nuestra API desarrollada en Python, el cual carga nuestro modelo **BOTNet_ConTuning.pth** y lo expone en Cloud Run para que pueda ser consumida en nuestro aplicativo web.

### Estructura del Proyecto

- **main.py:** Contiene el código de nuestra API donde tenemos la función `predict_anemia` y creamos el endpoint '/predict'.
- **requirements.txt:** Archivo plano con las librerías que utilizamos en nuestro proyecto.
- **Dockerfile:** Contenedor de nuestra API para luego ser expuesta en Cloud Run.

### Ejecución de la API en Local

Para ejecutar la API en local ejecuta el siguiente comando:

```bash
uvicorn main:app --reload --port 8000
```

### Desplegar API en Cloud Run

Primero creamos nuestro contenedor Docker con el siguiente comando:

```bash
gcloud builds submit --tag gcr.io/anemia-control-api/anemia-predict-api:v7
```

Luego, si todo salió bien en el paso anterior, desplegamos nuestro contenedor en Cloud Run para tener la API en línea ejecutando el siguiente comando:

```bash
gcloud run deploy anemia-control-api --image gcr.io/anemia-control-api/anemia-predict-api:v7 --platform managed --region us-central1 --allow-unauthenticated --port 8000 --cpu 1 --memory 2Gi --min-instances 0 --max-instances 1 --timeout 300s
```
