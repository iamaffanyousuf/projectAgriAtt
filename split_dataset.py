import os
import shutil
import random

SOURCE_DIR = "PlantVillage/PlantVillage"
TARGET_DIR = "dataset"

CLASSES = {
    "tomato_early_blight": "Tomato_Early_blight",
    "tomato_late_blight": "Tomato_Late_blight",
    "tomato_healthy": "Tomato_healthy",
}

SPLIT_RATIO = 0.8

for cls in CLASSES:
    os.makedirs(f"{TARGET_DIR}/train/{cls}", exist_ok=True)
    os.makedirs(f"{TARGET_DIR}/val/{cls}", exist_ok=True)

for target_cls, source_cls in CLASSES.items():
    src_path = os.path.join(SOURCE_DIR, source_cls)
    images = os.listdir(src_path)

    random.shuffle(images)
    split_idx = int(len(images) * SPLIT_RATIO)

    train_imgs = images[:split_idx]
    val_imgs = images[split_idx:]

    for img in train_imgs:
        shutil.copy(
            os.path.join(src_path, img), f"{TARGET_DIR}/train/{target_cls}/{img}"
        )

    for img in val_imgs:
        shutil.copy(os.path.join(src_path, img), f"{TARGET_DIR}/val/{target_cls}/{img}")

print("✅ Dataset split complete!")
