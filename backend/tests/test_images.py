import base64
import io
import json

import pytest
from PIL import Image

from studio import images, storage


@pytest.fixture(autouse=True)
def isolated_image_store(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "ROOT", tmp_path / "data")


def photo_url(image=None, format="PNG", **options):
    image = image if image is not None else Image.new("RGB", (24, 12), "red")
    output = io.BytesIO()
    image.save(output, format=format, **options)
    mime = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp", "GIF": "image/gif"}[format]
    return f"data:{mime};base64," + base64.b64encode(output.getvalue()).decode("ascii")


@pytest.mark.parametrize("format", ["JPEG", "PNG", "WEBP"])
def test_photo_round_trip_is_normalized_and_metadata_has_no_encoded_bytes(format):
    record = images.save_image("Factory line", photo_url(format=format))
    assert record == storage.get("reference", record["id"])
    assert record["mime_type"] == "image/jpeg"
    assert (record["width"], record["height"]) == (24, 12)
    assert record["created_at"]
    assert record["url"] == f"/api/references/{record['id']}/image"
    path = images.image_path(record["id"])
    assert path.parent == storage.ROOT / "references"
    assert record["size_bytes"] == path.stat().st_size
    assert "base64" not in json.dumps(record)
    encoded = images.image_data_url(record["id"])
    assert encoded.startswith("data:image/jpeg;base64,")
    assert base64.b64decode(encoded.split(",", 1)[1]) == path.read_bytes()
    with Image.open(path) as photo:
        assert photo.format == "JPEG"


def test_orientation_is_applied_metadata_stripped_and_large_photos_scaled():
    exif = Image.Exif()
    exif[274] = 6  # Rotate 90 degrees clockwise.
    exif[315] = "Private camera owner"
    record = images.save_image("Phone photo", photo_url(Image.new("RGB", (2000, 1000), "blue"), "JPEG", exif=exif))
    assert (record["width"], record["height"]) == (800, 1600)
    with Image.open(images.image_path(record["id"])) as photo:
        assert not photo.getexif()
        assert b"Private camera owner" not in images.image_path(record["id"]).read_bytes()


def test_transparency_is_composited_on_white():
    record = images.save_image("Transparent", photo_url(Image.new("RGBA", (16, 16), (0, 0, 0, 0))))
    with Image.open(images.image_path(record["id"])) as photo:
        assert photo.getpixel((8, 8)) == (255, 255, 255)


@pytest.mark.parametrize("data", [
    "", "https://example.com/photo.png", "data:image/png;base64,not-base64!",
    "data:image/png;base64,", "data:image/png;base64,\u00e9", "data:image/png;base64,YmFkIGltYWdl",
])
def test_malformed_photo_data_is_rejected_without_writing_images(data):
    with pytest.raises(ValueError):
        images.save_image("Invalid", data)
    assert not (storage.ROOT / "references").exists()


def test_unsupported_and_mislabeled_formats_are_rejected():
    gif = photo_url(format="GIF")
    for data in [gif, gif.replace("image/gif", "image/png"), photo_url().replace("image/png", "image/jpeg")]:
        with pytest.raises(ValueError):
            images.save_image("Unsupported", data)
    assert not (storage.ROOT / "references").exists()


@pytest.mark.parametrize("format", ["PNG", "WEBP"])
def test_animated_images_are_rejected(format):
    frames = [Image.new("RGB", (8, 8), color) for color in ("red", "blue")]
    encoded = photo_url(frames[0], format, save_all=True, append_images=frames[1:], duration=100, loop=0)
    with pytest.raises(ValueError, match="Animated"):
        images.save_image("Animation", encoded)


def test_oversized_encoded_payload_is_rejected_before_image_decode(monkeypatch):
    monkeypatch.setattr(images, "MAX_IMAGE_BYTES", 12)
    with pytest.raises(ValueError, match="8 MiB"):
        images.save_image("Too big", "data:image/png;base64," + "A" * 24)
    assert not (storage.ROOT / "references").exists()


def test_excess_pixel_count_is_rejected_before_loading(monkeypatch):
    data = photo_url()
    monkeypatch.setattr(images, "MAX_IMAGE_PIXELS", 100)
    def unexpected_load(*args, **kwargs):
        raise AssertionError("Oversized photos should be rejected before decoding pixels")
    monkeypatch.setattr(Image.Image, "load", unexpected_load)
    with pytest.raises(ValueError, match="20 megapixels"):
        images.save_image("Too many pixels", data)


def test_names_and_ids_cannot_escape_reference_storage(tmp_path):
    record = images.save_image("../../outside.jpg", photo_url())
    assert record["name"] == "outside.jpg"
    assert images.image_path(record["id"]).name == f"{record['id']}.jpg"
    assert not (tmp_path / "outside.jpg").exists()
    for invalid in ["../outside", "../../outside.jpg", "/etc/passwd", "f" * 33, "A" * 32, None]:
        with pytest.raises(KeyError):
            images.image_path(invalid)
    with pytest.raises(KeyError):
        images.image_path("f" * 32)
    images.image_path(record["id"]).unlink()
    with pytest.raises(KeyError):
        images.image_path(record["id"])


def test_symlink_cannot_move_photo_storage_outside_data_root(tmp_path):
    storage.ROOT.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (storage.ROOT / "references").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="storage directory"):
        images.save_image("photo", photo_url())
    assert not list(outside.iterdir())


def test_failed_metadata_write_does_not_leave_unregistered_photo(monkeypatch):
    def fail_save(*args, **kwargs):
        raise OSError("Database unavailable")
    monkeypatch.setattr(storage, "save", fail_save)
    with pytest.raises(OSError):
        images.save_image("Photo", photo_url())
    assert not list((storage.ROOT / "references").iterdir())
