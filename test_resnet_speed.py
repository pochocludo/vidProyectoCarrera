import time
import tensorflow as tf
from src.dataset import get_datasets
from src.models import build_model, compile_phase1

train_ds, val_ds, test_ds, class_names, class_weights = get_datasets(img_size=(256, 256), batch_size=32)

print("Building ResNet50...")
model, base_model = build_model("resnet50", num_classes=16, input_shape=(256, 256, 3))
compile_phase1(model)

print("Starting 1 test epoch of ResNet50 Phase 1...")
t0 = time.time()
model.fit(train_ds.take(5), epochs=1, verbose=1)
t1 = time.time()
print(f"5 batches took: {t1 - t0:.2f} s")
