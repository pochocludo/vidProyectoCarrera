import os
import sys

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

base_dir = r"C:\Users\jossu\OneDrive\Documentos\python\datasets\GLVD"

print("=" * 60)
print(f"Verificando estructura de: {base_dir}")
print("=" * 60)

for root, dirs, files in os.walk(base_dir):
    depth = root.replace(base_dir, '').count(os.sep)
    if depth <= 3:
        num_files = len(files)
        file_info = f"({num_files} archivos)" if num_files > 0 else ""
        print(f"{'  ' * depth}[DIR] {os.path.basename(root)} {file_info}")

print("\n" + "=" * 60)
print("Resumen de Clases de Hojas (Leaves):")
print("=" * 60)

leaves_dir = None
for root, dirs, _ in os.walk(base_dir):
    if os.path.basename(root).lower() == "leaves":
        leaves_dir = root
        break

if leaves_dir:
    for split in ["train", "val", "test"]:
        split_path = os.path.join(leaves_dir, split)
        if os.path.exists(split_path):
            classes = sorted(os.listdir(split_path))
            total_imgs = sum(len(os.listdir(os.path.join(split_path, c))) for c in classes)
            print(f"Split '{split}': {len(classes)} clases, {total_imgs} imagenes en total.")
            print(f"  Clases: {', '.join(classes[:8])}...")
            print(f"  Detalle por clase:")
            for c in classes:
                count = len(os.listdir(os.path.join(split_path, c)))
                print(f"    - {c}: {count} imgs")
else:
    print("No se encontró la carpeta 'Leaves' directamente. Estructura general:")
    for item in os.listdir(base_dir):
        print(f"  - {item}")
