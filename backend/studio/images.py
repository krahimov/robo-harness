"""Local photo references normalized for simulator authoring requests."""

import base64
import binascii
import io
import re
import uuid
import warnings
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

from . import storage as db

MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_IMAGES = 4
MAX_IMAGE_PIXELS = 20_000_000
MAX_IMAGE_SIDE = 1600
_FORMATS = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}
_DATA_URL = re.compile(r"^data:(image/(?:jpeg|png|webp));base64,")
_ID = re.compile(r"[0-9a-f]{32}\Z")


def _folder() -> Path:
    folder = db.ROOT / "references"
    # References remain beneath the configured data directory, including when
    # a local folder has accidentally been replaced with a symlink.
    if not folder.resolve().is_relative_to(db.ROOT.resolve()):
        raise ValueError("Photo storage directory is invalid")
    return folder


def _decode(data_url: str) -> tuple[str, bytes]:
    if not isinstance(data_url, str):
        raise ValueError("Provide a JPEG, PNG, or WebP photo as a base64 data URL")
    match = _DATA_URL.match(data_url)
    if match is None:
        raise ValueError("Provide a JPEG, PNG, or WebP photo as a base64 data URL")
    encoded = data_url[match.end():]
    if len(encoded) > 4 * ((MAX_IMAGE_BYTES + 2) // 3):
        raise ValueError("Photos must be 8 MiB or smaller")
    try:
        content = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error):
        raise ValueError("Photo data is not valid base64") from None
    if not content or len(content) > MAX_IMAGE_BYTES:
        raise ValueError("Photos must contain data and be 8 MiB or smaller")
    return match.group(1), content


def _normalize(mime_type: str, content: bytes) -> tuple[bytes, int, int]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(content)) as source:
                if source.format not in _FORMATS or _FORMATS[source.format] != mime_type:
                    raise ValueError("Photo contents must match their JPEG, PNG, or WebP type")
                if source.width * source.height > MAX_IMAGE_PIXELS:
                    raise ValueError("Photos must be 20 megapixels or smaller")
                if getattr(source, "n_frames", 1) != 1:
                    raise ValueError("Animated images are not supported; upload a still photo")
                source.load()
                oriented = ImageOps.exif_transpose(source)
                oriented.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE), Image.Resampling.LANCZOS)
                rgba = oriented.convert("RGBA")
                # Build a fresh image so EXIF, GPS, comments, and other source
                # metadata are never copied into the normalized photo.
                clean = Image.new("RGB", rgba.size, "white")
                clean.paste(rgba, mask=rgba.getchannel("A"))
                output = io.BytesIO()
                clean.save(output, format="JPEG", quality=88, optimize=True)
                return output.getvalue(), clean.width, clean.height
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise ValueError("Photos must be 20 megapixels or smaller") from None
    except (UnidentifiedImageError, OSError, SyntaxError):
        raise ValueError("Photo could not be decoded; upload a valid JPEG, PNG, or WebP image") from None


def save_image(name: str, data_url: str) -> dict:
    """Store a still photo and return public metadata without its encoded data."""
    mime_type, content = _decode(data_url)
    normalized, width, height = _normalize(mime_type, content)
    reference_id = uuid.uuid4().hex
    folder = _folder()
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{reference_id}.jpg"
    # Names are labels only. No user-supplied name participates in a disk path.
    label = re.split(r"[/\\]", name if isinstance(name, str) else "")[-1]
    label = "".join(char for char in label if char.isprintable()).strip()[:200] or "Photo"
    created = False
    try:
        with path.open("xb") as photo:
            created = True
            photo.write(normalized)
        return db.save("reference", {
            "id": reference_id,
            "name": label,
            "url": f"/api/references/{reference_id}/image",
            "width": width,
            "height": height,
            "mime_type": "image/jpeg",
            "size_bytes": len(normalized),
        })
    except Exception:
        if created:
            path.unlink(missing_ok=True)
        raise


def image_path(reference_id: str) -> Path:
    """Resolve an existing photo reference, never an arbitrary local path."""
    if not isinstance(reference_id, str) or _ID.fullmatch(reference_id) is None:
        raise KeyError("Photo reference not found")
    db.get("reference", reference_id)
    folder = _folder()
    path = folder / f"{reference_id}.jpg"
    if not path.resolve().is_relative_to(folder.resolve()) or not path.is_file():
        raise KeyError("Photo reference not found")
    return path


def image_data_url(reference_id: str) -> str:
    """Load a normalized reference for a server-side model request."""
    return "data:image/jpeg;base64," + base64.b64encode(image_path(reference_id).read_bytes()).decode("ascii")
