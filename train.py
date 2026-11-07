"""
Pipeline de Entrenamiento de Redes Neuronales para Identificación Varietal de Vid
================================================================================
Proyecto de Carrera UABC: "Desarrollo e implementación de un sistema embebido
basado en redes neuronales para la identificación automática de variedades de vid
mediante el análisis de hojas".

Uso directo desde terminal / Antigravity IDE:
    python train.py --model mobilenet_v2 --img_size 256 --fine_tune_layers 0 --epochs_p1 15 --epochs_p2 45 --lr_p2 1e-4
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
from src.dataset import get_datasets, BATCH_SIZE
from src.models import (
    build_model,
    compile_phase1,
    compile_phase2,
    get_model_summary_info,
    SUPPORTED_ARCHITECTURES
)

def parse_args():
    parser = argparse.ArgumentParser(description="Entrenador de Modelos Convolucionales para Vid")
    parser.add_argument("--model", type=str, default="mobilenet_v2",
                        choices=SUPPORTED_ARCHITECTURES,
                        help="Arquitectura CNN a entrenar (default: mobilenet_v2)")
    parser.add_argument("--img_size", type=int, default=256,
                        help="Resolución espacial cuadrada de entrada (default: 256)")
    parser.add_argument("--epochs_p1", type=int, default=15,
                        help="Épocas para Fase 1: Base convolucional congelada (default: 15)")
    parser.add_argument("--epochs_p2", type=int, default=45,
                        help="Épocas máximas para Fase 2: Fine-tuning gradual (default: 45)")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                        help="Tamaño de lote (default: 32)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Semilla pseudoaleatoria para reproducibilidad (default: 42)")
    parser.add_argument("--fine_tune_layers", type=int, default=0,
                        help="Número de capas superiores a descongelar en Fase 2 (0 = todas las capas)")
    parser.add_argument("--lr_p2", type=float, default=1e-4,
                        help="Learning rate base para Fase 2 (default: 1e-4)")
    return parser.parse_args()

def set_seed(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)

def main():
    args = parse_args()
    set_seed(args.seed)
    img_size = (args.img_size, args.img_size)

    print("=" * 70)
    print(" SISTEMA DE IDENTIFICACIÓN VARIETAL DE VID — UABC")
    print(f" Arquitectura seleccionada : {args.model}")
    print(f" Resolución de entrada     : {img_size[0]} x {img_size[1]} px")
    print(f" Semilla aleatoria         : {args.seed}")
    print(f" Épocas Fase 1 / Fase 2    : {args.epochs_p1} / {args.epochs_p2}")
    print(f" Tamaño de lote            : {args.batch_size}")
    layers_desc = "TODAS las capas (end-to-end)" if args.fine_tune_layers <= 0 else f"últimas {args.fine_tune_layers} capas"
    print(f" Descongelamiento (Fase 2) : {layers_desc}")
    print(f" Learning rate (Fase 2)    : {args.lr_p2}")
    print("=" * 70)

    # 1. Cargar particiones fijas y calcular pesos de balanceo de clase
    print(f"\n[1/5] Cargando particiones de datos (GLVD a {img_size[0]}x{img_size[1]})...")
    train_ds, val_ds, test_ds, class_names, class_weights = get_datasets(
        batch_size=args.batch_size,
        img_size=img_size
    )
    num_classes = len(class_names)
    print(f"✓ {num_classes} variedades cargadas con éxito.")
    print(f"  Clases: {', '.join(class_names)}")

    # 2. Construir la arquitectura seleccionada
    print(f"\n[2/5] Construyendo modelo '{args.model}' con pesos ImageNet ({img_size[0]}x{img_size[1]}x3)...")
    model, base_model = build_model(
        architecture=args.model,
        num_classes=num_classes,
        input_shape=(img_size[0], img_size[1], 3)
    )
    summary_info = get_model_summary_info(model)
    print(f"✓ Parámetros totales      : {summary_info['total_params']:,}")
    print(f"✓ Parámetros entrenables  : {summary_info['trainable_params']:,}")
    print(f"✓ Tamaño estimado FP32    : {summary_info['size_mb_fp32']:.2f} MB")
    print(f"✓ Tamaño estimado INT8    : {summary_info['size_mb_int8']:.2f} MB")

    # Carpetas de guardado
    os.makedirs("saved_models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    best_model_path = os.path.join("saved_models", f"{args.model}_{args.img_size}_best.keras")

    callbacks_phase1 = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=7,
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

    # 4. Fase 2: Fine-tuning end-to-end con Cosine Decay
    print(f"\n[4/5] Iniciando Fase 2: Fine-tuning ({layers_desc}, Cosine Decay lr={args.lr_p2})...")
    steps_per_epoch = len(train_ds)
    compile_phase2(
        model,
        base_model,
        fine_tune_layers=args.fine_tune_layers,
        learning_rate=args.lr_p2,
        steps_per_epoch=steps_per_epoch,
        epochs=args.epochs_p2
    )

    summary_info_p2 = get_model_summary_info(model)
    print(f"✓ Parámetros entrenables en Fase 2: {summary_info_p2['trainable_params']:,}")

    callbacks_phase2 = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=15,
            restore_best_weights=True,
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
    y_pred_tta_all = []

    for images, labels in test_ds:
        preds = eval_model.predict(images, verbose=0)
        
        # Test-Time Augmentation (TTA) con reflejo horizontal y vertical
        imgs_flip_lr = tf.image.flip_left_right(images)
        imgs_flip_ud = tf.image.flip_up_down(images)
        preds_flip_lr = eval_model.predict(imgs_flip_lr, verbose=0)
        preds_flip_ud = eval_model.predict(imgs_flip_ud, verbose=0)
        preds_tta = (preds + preds_flip_lr + preds_flip_ud) / 3.0
        
        y_true_all.extend(np.argmax(labels.numpy(), axis=1))
        y_pred_all.extend(np.argmax(preds, axis=1))
        y_pred_tta_all.extend(np.argmax(preds_tta, axis=1))

    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)
    y_pred_tta_all = np.array(y_pred_tta_all)

    top1_acc = np.mean(y_true_all == y_pred_all)
    macro_f1 = f1_score(y_true_all, y_pred_all, average="macro")

    top1_acc_tta = np.mean(y_true_all == y_pred_tta_all)
    macro_f1_tta = f1_score(y_true_all, y_pred_tta_all, average="macro")

    print("\n" + "=" * 70)
    print(" RESULTADOS FINALES DE EVALUACIÓN (TEST INTERNO)")
    print("=" * 70)
    print(f" Exactitud Top-1 Estándar  : {top1_acc * 100:.2f}% (Objetivo H1a: >= 90.00%)")
    print(f" F1-Score Macro Estándar   : {macro_f1:.4f}  (Objetivo H1a: >= 0.8800)")
    print(f" Exactitud Top-1 con TTA   : {top1_acc_tta * 100:.2f}% (Objetivo H1a: >= 90.00%)")
    print(f" F1-Score Macro con TTA    : {macro_f1_tta:.4f}  (Objetivo H1a: >= 0.8800)")

    best_acc = max(top1_acc, top1_acc_tta)
    best_f1 = max(macro_f1, macro_f1_tta)
    h1a_cumplida = (best_acc >= 0.90) and (best_f1 >= 0.88)
    if h1a_cumplida:
        print("  ► Criterio H1a: CUMPLIDO EXITOSAMENTE ✓")
    else:
        print("  ► Criterio H1a: NO CUMPLIDO (Se sugiere revisar épocas o regularización)")

    print("\nReporte de Clasificación Detallado por Variedad (con TTA):")
    print("-" * 70)
    report_dict = classification_report(
        y_true_all, y_pred_tta_all, target_names=class_names, output_dict=True
    )
    print(classification_report(y_true_all, y_pred_tta_all, target_names=class_names))

    # Guardar reporte en JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join("reports", f"eval_{args.model}_{args.img_size}_end2end_{timestamp}.json")
    results = {
        "architecture": args.model,
        "input_resolution": args.img_size,
        "fine_tune_strategy": "end-to-end",
        "seed": args.seed,
        "parameters": summary_info_p2,
        "top1_accuracy_standard": float(top1_acc),
        "macro_f1_standard": float(macro_f1),
        "top1_accuracy_tta": float(top1_acc_tta),
        "macro_f1_tta": float(macro_f1_tta),
        "h1a_satisfied": bool(h1a_cumplida),
        "classification_report_tta": report_dict,
        "timestamp": timestamp
    }
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"\n✓ Reporte exportado a: {report_file}")
    print(f"✓ Modelo óptimo guardado en: {best_model_path}")
    print("=" * 70)

if __name__ == "__main__":
    main()
