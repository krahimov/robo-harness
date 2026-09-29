import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(os.environ.get("STUDIO_DATA_DIR", Path(__file__).resolve().parents[2] / ".data"))


def now():
    return datetime.now(timezone.utc).isoformat()


def connection():
    ROOT.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(ROOT / "studio.sqlite", timeout=30)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("CREATE TABLE IF NOT EXISTS records (id TEXT PRIMARY KEY, kind TEXT NOT NULL, body TEXT NOT NULL)")
    return db


def save(kind, body):
    record = {"id": uuid.uuid4().hex, "created_at": now(), **body}
    with connection() as db:
        db.execute("INSERT INTO records VALUES (?, ?, ?)", (record["id"], kind, json.dumps(record, allow_nan=False)))
    return record


def save_many(items):
    """Publish linked records together, before any worker can act on them."""
    records = [(kind, {"id": uuid.uuid4().hex, "created_at": now(), **body}) for kind, body in items]
    with connection() as db:
        db.executemany("INSERT INTO records VALUES (?, ?, ?)",
            [(record["id"], kind, json.dumps(record, allow_nan=False)) for kind, record in records])
    return [record for _, record in records]


def get(kind, id):
    with connection() as db:
        row = db.execute("SELECT body FROM records WHERE kind=? AND id=?", (kind, id)).fetchone()
    if row is None:
        raise KeyError(f"{kind} not found")
    return json.loads(row[0])


def delete(kind, id):
    with connection() as db:
        db.execute("DELETE FROM records WHERE kind=? AND id=?", (kind, id))


def all_records(kind):
    with connection() as db:
        rows = db.execute("SELECT body FROM records WHERE kind=? ORDER BY rowid DESC", (kind,)).fetchall()
    return [json.loads(row[0]) for row in rows]


def update(kind, id, **changes):
    with connection() as db:
        db.execute("BEGIN IMMEDIATE")
        row = db.execute("SELECT body FROM records WHERE id=? AND kind=?", (id, kind)).fetchone()
        if row is None:
            raise KeyError(f"{kind} not found")
        record = {**json.loads(row[0]), **changes}
        db.execute("UPDATE records SET body=? WHERE id=? AND kind=?", (json.dumps(record, allow_nan=False), id, kind))
    return record


def write_artifact(data):
    id = uuid.uuid4().hex
    folder = ROOT / "artifacts"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{id}.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, separators=(",", ":"), allow_nan=False))
    temporary.replace(path)
    return id


def read_artifact(id):
    if len(id) != 32 or any(c not in "0123456789abcdef" for c in id):
        raise KeyError("Invalid artifact ID")
    path = ROOT / "artifacts" / f"{id}.json"
    if not path.exists():
        raise KeyError("Artifact not found")
    return json.loads(path.read_text())
