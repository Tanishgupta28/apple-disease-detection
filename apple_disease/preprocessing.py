"""One RGB decoder shared by auditing, training, and upload inference."""
from pathlib import Path
import warnings
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


def read_image(source, size=None):
    """Return uint8 RGB. Apply EXIF orientation before optional bilinear resize."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(source) as im:
                if im.width * im.height > 25_000_000:
                    raise ValueError("Image exceeds the 25-megapixel limit")
                im.load()
                im = ImageOps.exif_transpose(im).convert("RGB")
                if size is not None:
                    if not isinstance(size, int) or size < 8:
                        raise ValueError("Image size must be an integer >= 8")
                    im = im.resize((size, size), Image.Resampling.BILINEAR)
                return np.asarray(im).copy()
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as exc:
        raise ValueError("Cannot decode a safe image") from exc


def validate_rgb(image):
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 2:
        raise ValueError("Expected a nonempty H x W x 3 RGB image")
    if image.dtype != np.uint8:
        raise ValueError("Expected uint8 pixels in [0, 255]")
    return image
