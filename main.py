from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, HttpUrl
from PIL import Image
import io
import numpy as np
import torch
import timm
import torch.nn.functional as F
from torchvision import transforms
import requests
from fastapi.middleware.cors import CORSMiddleware

# --- Configuración del Modelo y Pre-procesamiento ---
MODEL_PATH = "BOTNet_ConTuning.pth"

# Inicializar la aplicación FastAPI
app = FastAPI(
    title="API de Detección de Anemia (BotNet Tuned)",
    description="API que recibe la URL de una imagen, la procesa y predice si el paciente es anémico o no utilizando un modelo BotNet con tuning.",
    version="1.0.1"
)

origins = [
    "http://localhost:3000",            # Aplicación frontend local
    "http://localhost",                 # Por si habilitamos otros puertos de localhost
    "https://anemia-control.web.app"    # Aplicación desplegada en Firebase Hosting
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,       # Lista de orígenes permitidos
    allow_credentials=True,      # Permitir cookies/credenciales
    allow_methods=["*"],         # Permitir todos los métodos (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],         # Permitir todas las cabeceras
)

# Variables globales para los modelos y el transformador
botnet_model= None
image_transform = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Funciones de Carga y Pre-procesamiento ---
@app.on_event("startup")
async def load_models_and_transforms():
    global botnet_model, image_transform

    print(f"Usando dispositivo: {device}")

    # Cargar el modelo BOTNet 
    try:
        botnet_model = timm.create_model("botnet26t_256", pretrained=False, num_classes=2)
        map_location = torch.device('cpu')
        botnet_model.load_state_dict(torch.load(MODEL_PATH, map_location=map_location))

        botnet_model.eval() # Poner el modelo en modo evaluación
        botnet_model.to(device) # Mover el modelo al dispositivo (GPU/CPU)

        print(f"Modelo BotNet desde '{MODEL_PATH}' cargado exitosamente y mapeado a '{device}'.")
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail=f"El archivo del modelo BotNet '{MODEL_PATH}' no se encontró.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al cargar el modelo BotNet: {e}. Asegúrate de que el modelo se haya guardado correctamente y sea compatible con la CPU. Detalle: {e}")

    # Definir la transformación de la imagen
    image_transform = transforms.Compose([
        transforms.Resize((256, 256)), # nueva resolución, antes era (224, 224)
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    print("Transformaciones de imagen inicializadas.")

# --- Clases Pydantic ---
# Para el request:
class ImageURL(BaseModel):
    url_image: HttpUrl

# Para el response: 
class AnemiaResponse(BaseModel):
    success: bool
    message: str
    clase: str | None = None
    confianza: str | None = None

# --- Endpoint para la predicción ---
@app.post("/predict/", response_model=AnemiaResponse)
async def predict_anemia(request: ImageURL):
    """
    Recibe la URL de una imagen en el request, la descarga y
    procesa con ViT y predice si el paciente es anémico o no.
    """
    if botnet_model is None or image_transform is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Los modelos o transformaciones no se cargaron correctamente al iniciar la API."
        )

    try:
        # Descargar la imagen desde la URL
        response = requests.get(str(request.url_image), stream=True)
        response.raise_for_status() # Lanza un error para códigos de estado HTTP 4xx/5xx

        # Abrir la imagen con PIL
        image = Image.open(io.BytesIO(response.content))
        if image.mode != 'RGB':
            image = image.convert("RGB")
        
        # --- Pre-procesar la imagen ---
        img_tensor = image_transform(image).unsqueeze(0) # Añadir dimensión de batch
        img_tensor = img_tensor.to(device) # Mover tensor a GPU/CPU

        # --- Realizar la predicción con el modelo BotNet ---
        with torch.no_grad():
            outputs = botnet_model(img_tensor) # Realizar la inferencia
            probabilities = F.softmax(outputs, dim=1) # Usar F.softmax
            pred_class = torch.argmax(probabilities, dim=1).item() # Obtener la clase predicha
            confidence = probabilities[0, pred_class].item() * 100 # Convertir a porcentaje

        # Determinar clase y precisión
        clases_map = {0: "No anémico", 1: "Anémico"}
        clase_text = clases_map.get(pred_class, "Desconocido") # Usar .get para seguridad

        confianza_value_float = confidence
        confianza_value = f"{confianza_value_float:.4f}" # Formatea como string con 4 decimales

        # Preparar la respuesta exitosa
        return AnemiaResponse(
            success=True,
            message="Prediction made successfully.",
            clase=clase_text,
            confianza=confianza_value
        )

    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, # Error del cliente: URL inválida/inaccesible
            detail=f"Error al descargar la imagen de la URL: {e}. Asegúrate de que la URL sea válida y accesible."
        )
    except Image.UnidentifiedImageError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, # Error del cliente: No es una imagen
            detail="El contenido de la URL no es una imagen válida o el formato no es soportado."
        )
    except Exception as e:
        print(f"Error inesperado durante la predicción: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, # Error interno del servidor
            detail=f"Error interno del servidor durante la predicción: {e}"
        )

# --- Endpoint de prueba/salud ---
@app.get("/")
async def health_check():
    """
    Endpoint para verificar si la API está funcionando.
    """
    return {"status": "ok", "message": "API de detección de anemia en funcionamiento."}