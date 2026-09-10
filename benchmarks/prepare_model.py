"""Export the official YOLO11n weights on CPU and record reproducible provenance.

Download source: https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt
Exporter docs: https://docs.ultralytics.com/modes/export/
Model artifacts stay in build/. No GPU or capture is initialized by this script.
"""
import argparse
import importlib.metadata
import json
from pathlib import Path
import urllib.request

from benchmark_contract import sha256

URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path(__file__).resolve().parents[1]/"build/section7/model")
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    weights = args.directory / "yolo11n.pt"
    if not weights.exists():
        temporary = weights.with_suffix(".download")
        urllib.request.urlretrieve(URL, temporary)
        temporary.replace(weights)
    from ultralytics import YOLO
    model = YOLO(str(weights))
    exported = Path(model.export(format="onnx", imgsz=640, batch=1,
                                  device="cpu", dynamic=False, simplify=False, opset=17))
    record = {"model":"YOLO11n", "trained":True, "weights_url":URL,
              "weights_sha256":sha256(weights), "onnx_sha256":sha256(exported),
              "onnx_path":str(exported.resolve()), "input":"RGB normalized, 1x3x640x640 FP32",
              "output":"raw detector predictions; no NMS or detection postprocessing timed",
              "export_options":{"imgsz":640,"batch":1,"device":"cpu","dynamic":False,"simplify":False,"opset":17},
              "packages":{name:importlib.metadata.version(name) for name in ("torch","ultralytics","onnx")}}
    exported.with_suffix(".provenance.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
