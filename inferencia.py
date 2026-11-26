"""Pipeline de inferencia para Faster R-CNN de torchvision."""
import cv2
import torch
from PIL import Image
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)
from torchvision.transforms import ToTensor


def cargar_modelo(checkpoint=None, clases=None):
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if checkpoint is None:
        pesos = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        modelo = fasterrcnn_resnet50_fpn(weights=pesos)
        clases = pesos.meta["categories"]
    else:
        if not clases or len(clases) < 2 or clases[0] != "__background__":
            raise ValueError("Indica las clases en orden, comenzando por __background__.")
        modelo = fasterrcnn_resnet50_fpn(
            weights=None, weights_backbone=None, num_classes=len(clases)
        )
        modelo.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    return modelo.to(dispositivo).eval(), clases, dispositivo


@torch.inference_mode()
def detectar(modelo, clases, dispositivo, imagenes, umbral=0.7):
    """Devuelve una lista de detecciones por imagen."""
    convertir = ToTensor()
    entradas = [convertir(imagen.convert("RGB")).to(dispositivo) for imagen in imagenes]
    resultados = []
    for salida in modelo(entradas):
        validas = salida["scores"] >= umbral
        resultados.append([
            {"clase": clases[etiqueta], "confianza": confianza, "caja": caja}
            for etiqueta, confianza, caja in zip(
                salida["labels"][validas].tolist(),
                salida["scores"][validas].tolist(),
                salida["boxes"][validas].tolist(),
            )
        ])
    return resultados


def detectar_video(modelo, clases, dispositivo, ruta, umbral=0.7, intervalo=5, limite=20):
    """Devuelve cada predicción conforme se procesa su fotograma."""
    if intervalo < 1 or limite < 1:
        raise ValueError("El intervalo y el límite deben ser positivos.")
    captura = cv2.VideoCapture(str(ruta))
    try:
        if not captura.isOpened():
            raise ValueError(f"No se pudo abrir el video: {ruta}")
        numero = 0
        muestras = 0
        while muestras < limite:
            disponible, frame = captura.read()
            if not disponible:
                break
            if numero % intervalo == 0:
                imagen = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                detecciones, = detectar(modelo, clases, dispositivo, [imagen], umbral)
                yield numero, imagen, detecciones
                muestras += 1
            numero += 1
        if muestras == 0:
            raise ValueError("El video no contiene fotogramas legibles.")
    finally:
        captura.release()
