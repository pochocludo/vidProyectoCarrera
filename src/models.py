import os
import tensorflow as tf
from tensorflow import keras
from keras import layers

SUPPORTED_ARCHITECTURES = [
    "mobilenet_v3_small",
    "mobilenet_v2",
    "efficientnet_b0",
    "resnet50"
]

def build_model(architecture="mobilenet_v2", num_classes=16, input_shape=(256, 256, 3), dropout_rate=0.35):
    """
    Construye una red neuronal convolucional mediante transferencia de aprendizaje
    a partir de pesos preentrenados en ImageNet.
    """
    arch = architecture.lower()
    if arch not in SUPPORTED_ARCHITECTURES:
        raise ValueError(f"Arquitectura '{arch}' no soportada. Opciones: {SUPPORTED_ARCHITECTURES}")

    inputs = layers.Input(shape=input_shape, name="imagen_entrada")

    # Preprocesamiento específico por arquitectura integrado en el modelo
    if arch == "mobilenet_v3_small":
        x = keras.applications.mobilenet_v3.preprocess_input(inputs)
        base_model = keras.applications.MobileNetV3Small(
            input_tensor=x,
            weights="imagenet",
            include_top=False,
            pooling=None
        )
    elif arch == "mobilenet_v2":
        x = keras.applications.mobilenet_v2.preprocess_input(inputs)
        base_model = keras.applications.MobileNetV2(
            input_tensor=x,
            weights="imagenet",
            include_top=False,
            pooling=None
        )
    elif arch == "efficientnet_b0":
        x = keras.applications.efficientnet.preprocess_input(inputs)
        base_model = keras.applications.EfficientNetB0(
            input_tensor=x,
            weights="imagenet",
            include_top=False,
            pooling=None
        )
    elif arch == "resnet50":
        x = keras.applications.resnet50.preprocess_input(inputs)
        base_model = keras.applications.ResNet50(
            input_tensor=x,
            weights="imagenet",
            include_top=False,
            pooling=None
        )

    # Congelar inicialmente la base convolucional para la Fase 1
    base_model.trainable = False

    # Cabezal clasificador robusto sin capas BN problemáticas
    features = base_model.output
    x = layers.GlobalAveragePooling2D(name="agrupamiento_promedio_global")(features)
    x = layers.Dropout(dropout_rate, name="dropout_1")(x)
    x = layers.Dense(256, activation="relu", name="capa_densa_1")(x)
    x = layers.Dropout(max(dropout_rate - 0.1, 0.15), name="dropout_2")(x)
    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32", name="probabilidades_variedad")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name=f"vid_{arch}")
    return model, base_model

def compile_phase1(model, learning_rate=1e-3):
    """
    Compilación para Fase 1: Base congelada, ajuste exclusivo del clasificador (Adam, lr=1e-3).
    """
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=[
            "accuracy",
            keras.metrics.TopKCategoricalAccuracy(k=3, name="top_3_accuracy")
        ]
    )
    return model

def compile_phase2(model, base_model, fine_tune_layers=0, learning_rate=1e-4,
                   steps_per_epoch=43, epochs=45):
    """
    Compilación para Fase 2: Descongelamiento y fine-tuning.
    Si fine_tune_layers <= 0, se descongela la base convolucional COMPLETA (end-to-end),
    manteniendo únicamente las capas BatchNormalization congeladas en modo inferencia
    para estabilizar estadísticas.
    """
    base_model.trainable = True
    total_layers = len(base_model.layers)
    
    if fine_tune_layers <= 0 or fine_tune_layers >= total_layers:
        cutoff = 0  # Descongelar TODAS las capas convolucionales
    else:
        cutoff = max(0, total_layers - fine_tune_layers)

    for layer in base_model.layers[:cutoff]:
        layer.trainable = False
    for layer in base_model.layers[cutoff:]:
        # Mantener BatchNorm congelado para preservar estadísticas de ImageNet
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True

    # Cosine Decay continuo
    total_steps = steps_per_epoch * epochs
    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=learning_rate,
        decay_steps=total_steps,
        alpha=0.01  # decae hasta el 1% del lr inicial (ej. 1e-6)
    )

    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=[
            "accuracy",
            keras.metrics.TopKCategoricalAccuracy(k=3, name="top_3_accuracy")
        ]
    )
    return model

def get_model_summary_info(model):
    """
    Retorna conteo de parámetros totales y entrenables.
    """
    total_params = model.count_params()
    trainable_params = sum(tf.size(w).numpy() for w in model.trainable_weights)
    non_trainable_params = total_params - trainable_params
    return {
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "non_trainable_params": int(non_trainable_params),
        "size_mb_fp32": total_params * 4 / (1024 * 1024),
        "size_mb_int8": total_params * 1 / (1024 * 1024)
    }

if __name__ == "__main__":
    print("=" * 60)
    print("Resumen de Arquitecturas Evaluadas:")
    print("=" * 60)
    for arch in SUPPORTED_ARCHITECTURES:
        m, b = build_model(arch, num_classes=16)
        info = get_model_summary_info(m)
        print(f"Arquitectura: {arch:<20} | Parámetros: {info['total_params']:,} | Tamaño est. INT8: {info['size_mb_int8']:.2f} MB")
