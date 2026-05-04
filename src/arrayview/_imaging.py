"""Shared lazy PIL accessors."""

_Image = None
_ImageOps = None


def ensure_image():
    """Lazy PIL.Image import — use wherever _pil_image() was previously defined."""
    global _Image
    if _Image is None:
        from PIL import Image

        _Image = Image
    return _Image


def ensure_imageops():
    """Lazy PIL.ImageOps import — use wherever _pil_imageops() was previously defined."""
    global _ImageOps
    if _ImageOps is None:
        from PIL import ImageOps

        _ImageOps = ImageOps
    return _ImageOps
