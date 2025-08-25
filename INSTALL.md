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

### `flash-attn`

`flash-attn` is not supported on macOS. Skip its installation or replace it with another attention implementation such as [`xformers`](https://github.com/facebookresearch/xformers).

### Handling `flash-attn` Installation Issues

macOS users can skip installing `flash_attn`.

If `flash-attn` fails due to **PEP 517 build issues**, you can try one of the following fixes.

#### No-Build-Isolation Installation (Recommended)
```bash
poetry run pip install --upgrade pip setuptools wheel
poetry run pip install flash-attn --no-build-isolation
poetry install
```

#### Install from Git (Alternative)
```bash
poetry run pip install git+https://github.com/Dao-AILab/flash-attention.git
```

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
