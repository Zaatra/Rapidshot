# INT8 dynamic quantization of YOLOX-S

Measured on the HP 324pe (Intel UHD) through ONNX Runtime's CPU provider with
intra-op spin-waiting disabled. `int8_plain` is `quantize_dynamic` on the
published model; `int8_prepared` runs `quant_pre_process` first (ONNX shape
inference and graph cleanup; symbolic shape inference skipped, as this model's
shapes are fully static).

Model sizes: FP32 35,858,002 bytes, int8_plain 9,082,271 bytes, int8_prepared 9,084,503 bytes.

Timings are the median of 8 runs on the dog fixture.

## 1 intra-op thread(s)

FP32: **232.2 ms** (min 218.2, max 243.8)

| Variant | p50 | Speed-up |
| --- | ---: | ---: |
| `int8_plain` | 121.4 ms | 1.91× |
| `int8_prepared` | 119.5 ms | 1.94× |

| Variant | Fixture | FP32 detections | INT8 detections | Within tolerance |
| --- | --- | ---: | ---: | --- |
| `int8_plain` | dog | 4 | 4 | **no** |
| `int8_plain` | wall | 0 | 0 | yes |
| `int8_plain` | moved_dog | 4 | 3 (classes [1, 2, 7, 16] vs [1, 2, 16]) | **no** |
| `int8_prepared` | dog | 4 | 4 | **no** |
| `int8_prepared` | wall | 0 | 0 | yes |
| `int8_prepared` | moved_dog | 4 | 3 (classes [1, 2, 7, 16] vs [1, 2, 16]) | **no** |

## 4 intra-op thread(s)

FP32: **68.2 ms** (min 65.9, max 69.4)

| Variant | p50 | Speed-up |
| --- | ---: | ---: |
| `int8_plain` | 62.3 ms | 1.09× |
| `int8_prepared` | 70.7 ms | 0.97× |

| Variant | Fixture | FP32 detections | INT8 detections | Within tolerance |
| --- | --- | ---: | ---: | --- |
| `int8_plain` | dog | 4 | 4 | **no** |
| `int8_plain` | wall | 0 | 0 | yes |
| `int8_plain` | moved_dog | 4 | 3 (classes [1, 2, 7, 16] vs [1, 2, 16]) | **no** |
| `int8_prepared` | dog | 4 | 4 | **no** |
| `int8_prepared` | wall | 0 | 0 | yes |
| `int8_prepared` | moved_dog | 4 | 3 (classes [1, 2, 7, 16] vs [1, 2, 16]) | **no** |

Tolerance is the frame-gate benchmark's: same class, boxes within 0.01 px, scores
within 0.00001.
