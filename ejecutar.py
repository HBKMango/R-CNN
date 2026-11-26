"""Ejecuta y mide el pipeline de la Actividad 4.1."""
import argparse
import json
import platform
from pathlib import Path
from statistics import mean, median
from time import perf_counter

import torch
from PIL import Image, ImageOps

from inferencia import cargar_modelo, detectar


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imagenes", type=Path, default=Path("datos"))
    parser.add_argument("--umbral", type=float, default=0.7)
    parser.add_argument("--repeticiones", type=int, default=3)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--clases", type=Path, help="JSON con nombres en orden de entrenamiento")
    args = parser.parse_args()

    if not 0 <= args.umbral <= 1 or args.repeticiones < 1:
        parser.error("El umbral debe estar entre 0 y 1 y las repeticiones deben ser positivas.")
    if bool(args.checkpoint) != bool(args.clases):
        parser.error("Indica --checkpoint y --clases juntos.")
    if not args.imagenes.is_dir():
        parser.error(f"No existe la carpeta {args.imagenes}.")
    rutas = sorted(p for p in args.imagenes.iterdir()
                   if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not rutas:
        parser.error(f"No hay imágenes JPG o PNG en {args.imagenes}.")

    imagenes = []
    for ruta in rutas:
        with Image.open(ruta) as imagen:
            imagenes.append(ImageOps.exif_transpose(imagen).convert("RGB"))
    nombres = json.loads(args.clases.read_text(encoding="utf-8")) if args.clases else None

    inicio = perf_counter()
    modelo, clases, dispositivo = cargar_modelo(args.checkpoint, nombres)
    carga = perf_counter() - inicio
    print(f"Modelo: Faster R-CNN ResNet50 FPN · pesos: {args.checkpoint or 'COCO'}")
    print(f"Arquitectura: {platform.machine()} · dispositivo: {dispositivo} · hilos: {torch.get_num_threads()}")
    print(f"Carga del modelo: {carga:.3f} s")
    print(f"Imágenes por repetición: {len(imagenes)} · umbral: {args.umbral:.0%}")

    def procesar():
        resultados = []
        for inicio_lote in range(0, len(imagenes), 2):
            lote = imagenes[inicio_lote:inicio_lote + 2]
            resultados.extend(detectar(modelo, clases, dispositivo, lote, args.umbral))
        return resultados

    procesar()
    print("Calentamiento completado.")
    tiempos = []
    for numero in range(1, args.repeticiones + 1):
        inicio = perf_counter()
        resultados = procesar()
        duracion = perf_counter() - inicio
        tiempos.append(duracion)
        print(f"Repetición {numero}: {duracion:.3f} s · {len(imagenes) / duracion:.2f} imágenes/s")

    for ruta, detecciones in zip(rutas, resultados):
        print(f"{ruta.name}: {len(detecciones)} detecciones")
        for deteccion in detecciones:
            caja = [round(coordenada, 1) for coordenada in deteccion["caja"]]
            print(f'  {deteccion["clase"]} · {deteccion["confianza"]:.2%} · caja {caja}')
    print(f"Tiempo medio: {mean(tiempos):.3f} s · mediana: {median(tiempos):.3f} s")
    print(f"Rendimiento global: {len(imagenes) * len(tiempos) / sum(tiempos):.2f} imágenes/s")


if __name__ == "__main__":
    main()
