# rapidshot-native

Prebuilt native GPU-interop extension for
[RapidShot](https://github.com/Zaatra/Rapidshot). Install it when you want the
GPU paths without installing a Rust toolchain:

```bash
pip install rapidshot rapidshot-native
```

That is the whole integration. This package contains no Python logic of its
own — RapidShot finds it automatically, and `rapidshot.native.is_available()`
starts returning `True`.

## What it adds

RapidShot's capture path is pure Python and works without this package.
The extension covers the parts Python cannot do:

- **The GPU tensor.** One compute dispatch turns a captured frame into a
  model-ready NCHW float32 tensor that never leaves VRAM, exposed as an
  `ID3D12Resource` for DirectML or as a shared NT handle for CUDA.
- **Cross-adapter transfer.** Moves a captured frame to a second GPU on hybrid
  laptops, where Desktop Duplication cannot capture from the discrete adapter
  at all — including the asynchronous path and its producer/consumer fence
  handshake.
- **AVX2 colour conversion.** Byte-exact against the NumPy path, and about
  6.4x faster on the synthetic conversion benchmark. Note that end-to-end this
  is worth closer to 1.1x on `grab()`, because the staging read dominates; the
  honest reason to install this package is the two items above.

Without it, RapidShot converts colour in NumPy and the GPU paths raise a clear
error naming what is missing. Nothing silently degrades.

## Requirements

- **Windows.** Desktop Duplication and D3D12 have no cross-platform equivalent,
  so there are no wheels for other platforms and there will not be.
- **Python 3.9+.** Built as an `abi3` wheel, so one binary covers every
  supported version — including versions of Python released after it was built.
- **x86-64.** AVX2 is detected at runtime, with scalar fallbacks, so the wheel
  runs on pre-AVX2 hardware.

## Versioning

Versioned independently of `rapidshot`: this wheel changes when the Rust
changes, which is not on every RapidShot release. `rapidshot` declares the
minimum it needs, so `pip` resolves a working pair. If you pin, pin both.

## Building from source

You do not need to — that is what this package is for — but if you want to:

```bash
cd native
cargo build --release
python install_dev.py
```

Needs [Rust](https://rustup.rs) 1.88+ and the MSVC C++ build tools. See
`RELEASING.md` in the repository for how the published wheels are produced.

## License

MIT, same as RapidShot.
