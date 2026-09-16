import hashlib
import io
import json
import sys

from PIL import Image, ImageCms


def decode_screenshot(data):
    image = Image.open(io.BytesIO(data))
    if image.format != "PNG" or image.width * image.height > 16_000_000:
        raise ValueError("Unsupported screenshot format or extent")
    profile = image.info.get("icc_profile")
    if not profile:
        raise ValueError("Screenshot color profile unavailable")
    source = ImageCms.ImageCmsProfile(io.BytesIO(profile))
    converted = ImageCms.profileToProfile(
        image, source, ImageCms.createProfile("sRGB"), outputMode="RGB",
        renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
    )
    return converted, {
        "width": image.width, "height": image.height,
        "sourceProfileSha256": hashlib.sha256(profile).hexdigest(),
        "sourceProfileDescription": ImageCms.getProfileDescription(source).strip(),
        "destination": "sRGB", "intent": "relative-colorimetric",
    }


if __name__ == "__main__":
    converted, metadata = decode_screenshot(sys.stdin.buffer.read())
    if sys.argv[1:] == ["--metadata"]:
        print(json.dumps(metadata))
    elif not sys.argv[1:]:
        sys.stdout.buffer.write(converted.tobytes())
    else:
        raise SystemExit("Usage: python tools/m109_pixels.py [--metadata] < screenshot.png")