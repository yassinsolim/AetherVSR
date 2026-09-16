import io

import pytest
from PIL import Image, ImageCms

from m109_pixels import decode_screenshot


def screenshot(profile=True):
    output = io.BytesIO()
    image = Image.new("RGB", (24, 16), (240, 40, 200))
    options = {"icc_profile": ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()} if profile else {}
    image.save(output, format="PNG", **options)
    return output.getvalue()


def test_profile_managed_color_and_identity():
    image, metadata = decode_screenshot(screenshot())
    assert image.size == (24, 16)
    assert image.getpixel((12, 8)) == (240, 40, 200)
    assert metadata["destination"] == "sRGB"
    assert metadata["intent"] == "relative-colorimetric"
    assert len(metadata["sourceProfileSha256"]) == 64


def test_unprofiled_screenshot_is_not_silent_srgb():
    with pytest.raises(ValueError, match="profile unavailable"):
        decode_screenshot(screenshot(False))


def test_invalid_image_is_not_evidence():
    with pytest.raises(Exception):
        decode_screenshot(b"not an image")