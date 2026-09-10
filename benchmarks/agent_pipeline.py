"""Present submission -> PNG/JPEG base64 data URL; no network request is sent."""
import base64
import io


def encode(rgb, codec="png", quality=90):
    from PIL import Image
    stream = io.BytesIO()
    Image.fromarray(rgb).save(stream, format=codec.upper(),
                             **({"quality": quality} if codec == "jpeg" else {}))
    payload = stream.getvalue()
    return f"data:image/{codec};base64," + base64.b64encode(payload).decode("ascii"), len(payload)


if __name__ == "__main__":
    from section7 import main
    raise SystemExit(main("agent"))
