# Installation Guide

## Install with pip

```bash
pip install .
pip install .[dev]  # Installe aussi les outils de dev
```

## Install with Poetry

Ensure you have [Poetry](https://python-poetry.org/docs/#installation) installed on your system.

To install all dependencies:

```bash
poetry install
```

## macOS

### PyTorch with MPS

Install PyTorch with [MPS](https://pytorch.org/docs/stable/mps.html) support for Apple Silicon or Intel Macs:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### `ffmpeg` via Homebrew

Use [Homebrew](https://brew.sh) to install `ffmpeg` for video generation:

```bash
brew install ffmpeg
```

### Install project dependencies

After installing PyTorch, install the remaining dependencies. The requirements
file installs `xformers` and automatically skips `flash-attn` on macOS:

```bash
pip install -r requirements.txt
```

`flash-attn` is currently unsupported on macOS, but `xformers` provides similar
attention optimizations for Apple hardware.

When the MPS backend is available, `generate.py` automatically sets
`PYTORCH_ENABLE_MPS_FALLBACK=1` to allow unsupported operations to fall back to
the CPU. To override this default, define the variable manually before running
the script:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1  # optional manual override
```

`generate.py` also defaults `TORCH_MPS_HIGH_WATERMARK_RATIO` to `0.8` to balance
MPS memory usage. Lower ratios free memory back to macOS more aggressively,
reducing OOM risk at the cost of performance, while higher ratios keep more
memory cached for speed but may exhaust available resources. Override the
default with `--mps-watermark <ratio>` or by setting the environment variable
manually.

---

### Running the Model

Once the installation is complete, you can run **Wan2.2** using:

```bash
poetry run python generate.py --task t2v-A14B --size '1280*720' --ckpt_dir ./Wan2.2-T2V-A14B --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
```

On machines with limited GPU support (e.g. macOS), you can offload model components to the CPU:

```bash
poetry run python generate.py --task t2v-A14B --size '512*512' --ckpt_dir ./Wan2.2-T2V-A14B --prompt "Two anthropomorphic cats" --offload_model true
```

#### Test
```bash
bash tests/test.sh
```

#### Format
```bash
black .
isort .
```
