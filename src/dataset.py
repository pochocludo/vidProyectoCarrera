import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.utils.class_weight import compute_class_weight

# Localización predeterminada del dataset dentro del proyecto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "datasets", "GLVD", "Grapevine Leaf Variety & Disease Dataset (GLVD)", "Leaves")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

def get_data_augmentation():
    """
    Aumento de datos aplicado EXCLUSIVAMENTE sobre la partición de entrenamiento.
    Cumple con las especificaciones de la Tabla 3.8 del documento de tesis.
    """
    return keras.Sequential([
        layers.RandomRotation(factor=0.1, fill_mode="reflect", name="rotacion_aleatoria"),
        layers.RandomFlip(mode="horizontal_and_vertical", name="reflejo_aleatorio"),
        layers.RandomZoom(height_factor=0.1, width_factor=0.1, fill_mode="reflect", name="zoom_aleatorio"),
        layers.RandomBrightness(factor=0.1, name="brillo_aleatorio"),
        layers.RandomContrast(factor=0.1, name="contraste_aleatorio"),
    ], name="data_augmentation")

def compute_weights(train_dir, class_names):
    """
    Calcula pesos inversos por clase para mitigar el desbalance muestral (Sección 2.3.5 de la tesis).
    w_k = N / (K * N_k)
    """
    y_train = []
    for idx, cname in enumerate(class_names):
        folder = os.path.join(train_dir, cname)
        imgs = glob.glob(os.path.join(folder, "*.*"))
        y_train.extend([idx] * len(imgs))
        
    y_train = np.array(y_train)
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight_dict = {cls_idx: float(w) for cls_idx, w in zip(classes, weights)}
    return class_weight_dict

def load_split(split_dir, shuffle=True, batch_size=BATCH_SIZE, img_size=IMG_SIZE):
    """
    Carga una partición de imágenes usando image_dataset_from_directory.
    Las etiquetas se retornan en formato categórico (one-hot).
    """
    if not os.path.exists(split_dir):
        raise FileNotFoundError(f"No se encontró el directorio de la partición: {split_dir}")
        
    ds = tf.keras.utils.image_dataset_from_directory(
        split_dir,
        labels="inferred",
        label_mode="categorical",
        image_size=img_size,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=42,
        interpolation="bicubic"
    )
    return ds

def get_datasets(dataset_path=None, batch_size=BATCH_SIZE, img_size=IMG_SIZE, augment_train=True):
    """
    Carga y prepara las tres particiones fijas (train, val, test) del dataset GLVD.
    Aplica Data Augmentation únicamente en el conjunto de entrenamiento.
    """
    if dataset_path is None:
        dataset_path = DATASET_DIR
        
    train_dir = os.path.join(dataset_path, "train")
    val_dir = os.path.join(dataset_path, "val")
    test_dir = os.path.join(dataset_path, "test")
    
    # Cargar datasets
    train_ds = load_split(train_dir, shuffle=True, batch_size=batch_size, img_size=img_size)
    val_ds = load_split(val_dir, shuffle=False, batch_size=batch_size, img_size=img_size)
    test_ds = load_split(test_dir, shuffle=False, batch_size=batch_size, img_size=img_size)
    
    class_names = train_ds.class_names
    class_weights = compute_weights(train_dir, class_names)
    
    # Data Augmentation en entrenamiento
    if augment_train:
        data_aug = get_data_augmentation()
        train_ds = train_ds.map(lambda x, y: (data_aug(x, training=True), y),
                                num_parallel_calls=tf.data.AUTOTUNE)
        
    # Optimización de pipeline de E/S
    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return train_ds, val_ds, test_ds, class_names, class_weights

if __name__ == "__main__":
    print("=" * 60)
    print("Verificando módulo de carga de datos...")
    print(f"Ruta: {DATASET_DIR}")
    print("=" * 60)
    
    train_ds, val_ds, test_ds, classes, weights = get_datasets()
    print(f"Total de clases detectadas: {len(classes)}")
    print(f"Clases: {classes}")
    print("\nPesos por clase calculados:")
    for idx, name in enumerate(classes):
        print(f"  {name:20s}: {weights[idx]:.4f}")
