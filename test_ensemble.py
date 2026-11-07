import os
import sys
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, f1_score

sys.stdout.reconfigure(encoding='utf-8')

from src.dataset import get_datasets

_, _, test_ds_224, class_names, _ = get_datasets(img_size=(224, 224))
_, _, test_ds_256, _, _ = get_datasets(img_size=(256, 256))

model_224 = tf.keras.models.load_model("saved_models/mobilenet_v2_best.keras")
model_256 = tf.keras.models.load_model("saved_models/mobilenet_v2_256_best.keras")

preds_224_list = []
y_true = []
for imgs, labels in test_ds_224:
    p = model_224.predict(imgs, verbose=0)
    p_flip = model_224.predict(tf.image.flip_left_right(imgs), verbose=0)
    preds_224_list.extend(0.5 * (p + p_flip))
    y_true.extend(np.argmax(labels.numpy(), axis=1))

preds_256_list = []
for imgs, labels in test_ds_256:
    p = model_256.predict(imgs, verbose=0)
    p_lr = model_256.predict(tf.image.flip_left_right(imgs), verbose=0)
    p_ud = model_256.predict(tf.image.flip_up_down(imgs), verbose=0)
    preds_256_list.extend((p + p_lr + p_ud) / 3.0)

preds_224_arr = np.array(preds_224_list)
preds_256_arr = np.array(preds_256_list)
y_true = np.array(y_true)

for w256 in [0.5, 0.6, 0.7, 0.8]:
    p_ens = w256 * preds_256_arr + (1.0 - w256) * preds_224_arr
    y_ens = np.argmax(p_ens, axis=1)
    acc = np.mean(y_true == y_ens)
    f1 = f1_score(y_true, y_ens, average='macro')
    print(f"Ensemble (w256={w256:.2f}): Top-1 Acc = {acc*100:.2f}%, Macro F1 = {f1:.4f}")
