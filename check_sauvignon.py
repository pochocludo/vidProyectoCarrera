import os, sys, glob
import numpy as np
import tensorflow as tf

sys.stdout.reconfigure(encoding='utf-8')
from src.dataset import get_datasets

_, _, test_ds_256, class_names, _ = get_datasets(img_size=(256, 256))
model_256 = tf.keras.models.load_model("saved_models/mobilenet_v2_256_best.keras")

sb_idx = class_names.index("Sauvignon Blanc")
print(f"Sauvignon Blanc class index: {sb_idx}")

y_true = []
all_preds = []
for imgs, labels in test_ds_256:
    p = model_256.predict(imgs, verbose=0)
    all_preds.extend(p)
    y_true.extend(np.argmax(labels.numpy(), axis=1))

y_true = np.array(y_true)
all_preds = np.array(all_preds)

sb_samples = np.where(y_true == sb_idx)[0]
print("Sauvignon Blanc test samples indices:", sb_samples)

for s in sb_samples:
    probs = all_preds[s]
    top3_idx = np.argsort(probs)[::-1][:3]
    top3_str = ", ".join([f"{class_names[i]}: {probs[i]:.3f}" for i in top3_idx])
    print(f"Sample {s}: Top-3 -> {top3_str} (True={class_names[y_true[s]]}, P_sb={probs[sb_idx]:.3f})")
