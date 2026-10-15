"""
Pipeline de Entrenamiento de Redes Neuronales para Identificación Varietal de Vid
================================================================================
Proyecto de Carrera UABC: "Desarrollo e implementación de un sistema embebido
basado en redes neuronales para la identificación automática de variedades de vid
mediante el análisis de hojas".

Uso directo desde terminal / Antigravity IDE:
    python train.py --model mobilenet_v3_small --epochs_p1 10 --epochs_p2 20
    python train.py --model mobilenet_v2
    python train.py --model efficientnet_b0
    python train.py --model resnet50
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
import numpy as np

# Configurar salida UTF-8 para consola en Windows
sys.stdout.reconfigure(encoding='utf-8')

# Importar TensorFlow y Keras
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# Importar módulos locales
from src.dataset import get_datasets, IMG_SIZE, BATCH_SIZE
from src.models import (
    build_model,
    compile_phase1,
    compile_phase2,
    get_model_summary_info,
    SUPPORTED_ARCHITECTURES
)

def parse_args():
    parser = argparse.ArgumentParser(description="Entrenador de Modelos Convolucionales para Vid")
    parser.add_argument("--model", type=str, default="mobilenet_v3_small",
                        choices=SUPPORTED_ARCHITECTURES,
                        help="Arquitectura CNN a entrenar (default: mobilenet_v3_small)")
    parser.add_argument("--epochs_p1", type=int, default=10,
                        help="Épocas para Fase 1: Base convolucional congelada (default: 10)")
    parser.add_argument("--epochs_p2", type=int, default=20,
                        help="Épocas máximas para Fase 2: Fine-tuning gradual (default: 20)")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                        help="Tamaño de lote (default: 32)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Semilla pseudoaleatoria para reproducibilidad (default: 42)")
    parser.add_argument("--fine_tune_layers", type=int, default=30,
                        help="Número de capas superiores a descongelar en Fase 2 (default: 30)")
    return parser.parse_args()

def set_seed(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)

def main():
    args = parse_args()
    set_seed(args.seed)

    print("=" * 70)
    print(" SISTEMA DE IDENTIFICACIÓN VARIETAL DE VID — UABC")
    print(f" Arquitectura seleccionada : {args.model}")
    print(f" Semilla aleatoria         : {args.seed}")
    print(f" Épocas Fase 1 / Fase 2    : {args.epochs_p1} / {args.epochs_p2}")
    print(f" Tamaño de lote            : {args.batch_size}")
    print("=" * 70)

    # 1. Cargar particiones fijas y calcular pesos de balanceo de clase
    print("\n[1/5] Cargando particiones de datos (GLVD)...")
    train_ds, val_ds, test_ds, class_names, class_weights = get_datasets(batch_size=args.batch_size)
    num_classes = len(class_names)
    print(f"✓ {num_classes} variedades cargadas con éxito.")
    print(f"  Clases: {', '.join(class_names)}")

    # 2. Construir la arquitectura seleccionada
    print(f"\n[2/5] Construyendo modelo '{args.model}' con pesos ImageNet...")
    model, base_model = build_model(
        architecture=args.model,
        num_classes=num_classes,
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )
    summary_info = get_model_summary_info(model)
    print(f"✓ Parámetros totales      : {summary_info['total_params']:,}")
    print(f"✓ Tamaño estimado FP32    : {summary_info['size_mb_fp32']:.2f} MB")
    print(f"✓ Tamaño estimado INT8    : {summary_info['size_mb_int8']:.2f} MB")

    # Carpetas de guardado
    os.makedirs("saved_models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    best_model_path = os.path.join("saved_models", f"{args.model}_best.keras")

    callbacks_phase1 = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
    ]

    # 3. Fase 1: Entrenamiento con base congelada
    print("\n[3/5] Iniciando Fase 1: Ajuste del clasificador (base congelada, lr=1e-3)...")
    compile_phase1(model, learning_rate=1e-3)
    t0 = time.time()
    history_p1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs_p1,
        class_weight=class_weights,
        callbacks=callbacks_phase1,
        verbose=1
    )
    t_p1 = time.time() - t0
    print(f"✓ Fase 1 completada en {t_p1:.1f} segundos.")

    # 4. Fase 2: Fine-tuning gradual
    print(f"\n[4/5] Iniciando Fase 2: Fine-tuning (últimas {args.fine_tune_layers} capas, lr=1e-5)...")
    compile_phase2(
        model,
        base_model,
        fine_tune_layers=args.fine_tune_layers,
        learning_rate=1e-5
    )

    callbacks_phase2 = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=best_model_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        )
    ]

    t0 = time.time()
    history_p2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs_p2,
        class_weight=class_weights,
        callbacks=callbacks_phase2,
        verbose=1
    )
    t_p2 = time.time() - t0
    print(f"✓ Fase 2 completada en {t_p2:.1f} segundos.")

    # 5. Evaluación en partición de prueba interna (Contraste H1a)
    print("\n[5/5] Evaluando en la partición de prueba interna...")
    # Cargar mejor modelo si fue guardado
    if os.path.exists(best_model_path):
        eval_model = keras.models.load_model(best_model_path)
    else:
        eval_model = model

    y_true_all = []
    y_pred_all = []

    for images, labels in test_ds:
        preds = eval_model.predict(images, verbose=0)
        y_true_all.extend(np.argmax(labels.numpy(), axis=1))
        y_pred_all.extend(np.argmax(preds, axis=1))

    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)

    top1_acc = np.mean(y_true_all == y_pred_all)
    macro_f1 = f1_score(y_true_all, y_pred_all, average="macro")

    print("\n" + "=" * 70)
    print(" RESULTADOS FINALES DE EVALUACIÓN (TEST INTERNO)")
    print("=" * 70)
    print(f" Exactitud Top-1 (Accuracy) : {top1_acc * 100:.2f}% (Objetivo H1a: >= 90.00%)")
    print(f" F1-Score Macro             : {macro_f1:.4f}  (Objetivo H1a: >= 0.8800)")

    h1a_cumplida = (top1_acc >= 0.90) and (macro_f1 >= 0.88)
    if h1a_cumplida:
        print("  ► Criterio H1a: CUMPLIDO EXITOSAMENTE ✓")
    else:
        print("  ► Criterio H1a: NO CUMPLIDO (Se sugiere revisar épocas o regularización)")

    print("\nReporte de Clasificación Detallado por Variedad:")
    print("-" * 70)
    report_dict = classification_report(
        y_true_all, y_pred_all, target_names=class_names, output_dict=True
    )
    print(classification_report(y_true_all, y_pred_all, target_names=class_names))

    # Guardar reporte en JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join("reports", f"eval_{args.model}_{timestamp}.json")
    results = {
        "architecture": args.model,
        "seed": args.seed,
        "parameters": summary_info,
        "top1_accuracy": float(top1_acc),
        "macro_f1": float(macro_f1),
        "h1a_satisfied": bool(h1a_cumplida),
        "classification_report": report_dict,
        "timestamp": timestamp
    }
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"\n✓ Reporte exportado a: {report_file}")
    print(f"✓ Modelo óptimo guardado en: {best_model_path}")
    print("=" * 70)

if __name__ == "__main__":
    main()
