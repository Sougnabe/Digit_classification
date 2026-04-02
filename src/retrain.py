import shutil
from pathlib import Path

from src.upload_store import materialize_uploaded_files_to_train


UPLOAD_DIR = Path("data/uploads")
TRAIN_DIR = Path("data/train")


ALLOWED_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".gif"}


def move_uploaded_data_into_train() -> int:
    moved = 0
    if not UPLOAD_DIR.exists():
        return moved

    for class_dir in UPLOAD_DIR.iterdir():
        if not class_dir.is_dir():
            continue

        target_class_dir = TRAIN_DIR / class_dir.name
        target_class_dir.mkdir(parents=True, exist_ok=True)

        for item in class_dir.iterdir():
            if item.is_file() and item.suffix.lower() in ALLOWED_SUFFIXES:
                # Put uploaded image into the right class folder for retraining.
                shutil.move(str(item), str(target_class_dir / item.name))
                moved += 1

    return moved


def retrain_model(epochs: int = 3):
    from src.model import train_and_evaluate

    moved_files = move_uploaded_data_into_train()
    copied_files = materialize_uploaded_files_to_train()
    metrics = train_and_evaluate(epochs=epochs)
    metrics["moved_files"] = moved_files
    metrics["copied_files"] = copied_files
    return metrics


if __name__ == "__main__":
    output = retrain_model(epochs=3)
    print("Retraining complete")
    print(output)
