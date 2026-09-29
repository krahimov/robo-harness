"""Private SSH-tunneled studio, including its prebuilt frontend."""
from pathlib import Path
import json
from fastapi.staticfiles import StaticFiles
from studio.main import app
from studio import storage as db

ROOT = Path(__file__).resolve().parents[2]


@app.get('/api/cloud/trial')
def trial_receipt():
    return json.loads((db.ROOT/'trial-receipt.json').read_text())


app.mount('/', StaticFiles(directory=ROOT/'web/dist', html=True), name='studio-ui')
