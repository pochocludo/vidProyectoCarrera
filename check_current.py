import os
import sys
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

sys.stdout.reconfigure(encoding='utf-8')

from src.dataset import get_datasets

train_ds, val_ds, test_ds, class_names, class_weights = get_datasets(img_size=(256, 256))

model_path = "saved_models/mobilenet_v2_256_best.keras"
if not os.path.exists(model_path):
    print("Model not found:", model_path)
    sys.exit(1)

model = tf.keras.models.load_model(model_path)

y_true = []
y_pred_std = []
y_pred_tta = []

for imgs, labels in test_ds:
    p_std = model.predict(imgs, verbose=0)
    p_flip_lr = model.predict(tf.image.flip_left_right(imgs), verbose=0)
    p_flip_ud = model.predict(tf.image.flip_up_down(imgs), verbose=0)
    p_tta = (p_std + p_flip_lr + p_flip_ud) / 3.0
    
    y_true.extend(np.argmax(labels.numpy(), axis=1))
    y_pred_std.extend(np.argmax(p_std, axis=1))
    y_pred_tta.extend(np.argmax(p_tta, axis=1))

y_true = np.array(y_true)
y_pred_std = np.array(y_pred_std)
y_pred_tta = np.array(y_pred_tta)

print(f"Standard Test Acc: {np.mean(y_true == y_pred_std)*100:.2f}%")
print(f"3-way TTA Test Acc: {np.mean(y_true == y_pred_tta)*100:.2f}%")

print("\nMisclassifications with 3-way TTA:")
for i in range(len(y_true)):
    if y_true[i] != y_pred_tta[i]:
        print(f"Sample {i}: True={class_names[y_true[i]]} -> Pred={class_names[y_pred_tta[i]]}")
