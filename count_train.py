import os, glob

base = os.path.join("datasets", "GLVD", "Grapevine Leaf Variety & Disease Dataset (GLVD)", "Leaves", "train")
for c in sorted(os.listdir(base)):
    cnt = len(glob.glob(os.path.join(base, c, "*.*")))
    print(f"{c:22s}: {cnt:4d}")
