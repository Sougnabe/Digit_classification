import sqlite3
from datetime import datetime, timezone
from pathlib import Path


UPLOAD_DB_PATH = Path("data/uploads/uploads.sqlite3")


def get_connection():
    UPLOAD_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(UPLOAD_DB_PATH)
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS uploaded_images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            class_name TEXT NOT NULL,
            filename TEXT NOT NULL,
            content_type TEXT,
            content BLOB NOT NULL,
            created_at TEXT NOT NULL,
            processed INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    connection.commit()
    return connection


def save_uploaded_file(class_name: str, filename: str, content: bytes, content_type: str | None = None) -> int:
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO uploaded_images (class_name, filename, content_type, content, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (class_name, filename, content_type, content, created_at),
        )
        connection.commit()
        return int(cursor.lastrowid)


def materialize_uploaded_files_to_train(train_dir: str = "data/train") -> int:
    train_path = Path(train_dir)
    train_path.mkdir(parents=True, exist_ok=True)

    if not UPLOAD_DB_PATH.exists():
        return 0

    copied = 0
    with get_connection() as connection:
        rows = connection.execute(
            "SELECT id, class_name, filename, content FROM uploaded_images WHERE processed = 0 ORDER BY id"
        ).fetchall()

    for row_id, class_name, filename, content in rows:
        class_dir = train_path / str(class_name)
        class_dir.mkdir(parents=True, exist_ok=True)
        out_path = class_dir / f"db_{row_id}_{filename}"
        out_path.write_bytes(content)
        copied += 1

    if rows:
        with get_connection() as connection:
            connection.executemany(
                "UPDATE uploaded_images SET processed = 1 WHERE id = ?",
                [(row_id,) for row_id, _, _, _ in rows],
            )
            connection.commit()

    return copied