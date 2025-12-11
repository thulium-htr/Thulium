# Thulium – Multilingual Handwriting Intelligence for Python

[![Build Status](https://img.shields.io/github/actions/workflow/status/olaflaitinen/Thulium/ci.yml?branch=main)](https://github.com/olaflaitinen/Thulium/actions)
[![Coverage](https://img.shields.io/codecov/c/github/olaflaitinen/Thulium)](https://codecov.io/gh/olaflaitinen/Thulium)
[![PyPI version](https://img.shields.io/pypi/v/thulium)](https://pypi.org/project/thulium/)
[![Python Versions](https://img.shields.io/pypi/pyversions/thulium)](https://pypi.org/project/thulium/)
[![License](https://img.shields.io/github/license/olaflaitinen/Thulium)](LICENSE)

**Thulium** is a state-of-the-art, open-source library for offline handwritten text recognition (HTR) and document intelligence. Designed for high-performance research and production use cases, Thulium provides an end-to-end stack—from document layout analysis to language-model-enhanced decoding-for over **50 languages**, with deep support for **Azerbaijani**, English, Turkish, and other major global scripts.

> **Note**: Thulium is currently in alpha status. APIs are subject to change.

## Overview

Thulium abstracts the complexity of modern deep learning-based OCR/HTR pipelines into a clean, comprehensive Python API. Whether you are digitizing historical archives, processing structured forms, or building reading systems for low-resource languages, Thulium provides the architectural flexibility and performance you need.

### Key Capabilities

*   **Multilingual Deep Learning**: Architecture supports pluggable language profiles. Out-of-the-box support planned for Latin, Cyrillic, Arabic, and Indic scripts.
*   **End-to-End Pipeline**: Full coverage including:
    *   **Preprocessing**: Image normalization, binarization, and augmentation.
    *   **Segmentation**: Robust line and word segmentation (U-Net based).
    *   **Recognition**: CNN-RNN-CTC and Transformer-based HTR models.
    *   **Decoding**: Greedy, Beam Search, and LM-enhanced decoding.
*   **Corporate & Academic Design**: Built with modularity, extensibility, and rigorous testing in mind.
*   **Explainability (XAI)**: Built-in tools for attention mapping and error analysis.

## Installation

### From PyPI (Coming Soon)

```bash
pip install thulium
```

### From Source (Development)

To install Thulium for development or research purposes:

```bash
git clone https://github.com/olaflaitinen/Thulium.git
cd Thulium
pip install -e .[dev]
```

## Quickstart

### Python API

The high-level API automates model selection and pipeline orchestration.

```python
from thulium.api import recognize_image

# Recognize text in an Azerbaijani document
result = recognize_image(
    path="docs/samples/handwriting.jpg",
    language="az",
    device="auto"  # Automatically uses GPU if available
)

print(f"Full Text:\n{result.full_text}")

# Inspect confidence per line
for line in result.lines:
    if line.confidence < 0.8:
        print(f"Low confidence line [{line.confidence:.2f}]: {line.text}")
```

### CLI Usage

Thulium includes a robust Command Line Interface (CLI) for batch processing and evaluation.

```bash
# Basic recognition
thulium recognize my_document.scanned.jpg --language az --output result.json

# Verbose logging
thulium recognize page_01.png -l en -v
```

## Architecture

Thulium is organized into modular components to facilitate research and extension:

| Module | Description |
| :--- | :--- |
| `thulium.api` | High-level entry points for ease of use. |
| `thulium.data` | Loaders, transforms, and dataset abstractions. |
| `thulium.models` | PyTorch implementations of backbones, heads, and decoders. |
| `thulium.pipeline` | Logic for chaining segmentation and recognition steps. |
| `thulium.evaluation` | Metrics (CER, WER) and benchmarking tools. |

For a deep dive, see the [Architecture Documentation](docs/architecture.md).

## Language Support

Thulium is architected to support **50+ languages**.

| Region | Languages |
| :--- | :--- |
| **Middle East & Central Asia** | **Azerbaijani**, Turkish, Arabic, Persian, Urdu |
| **Western Europe** | English, German, French, Spanish, Italian, Dutch, Portuguese |
| **Eastern Europe** | Russian, Ukrainian, Polish, Czech, Hungarian |
| **Scandinavia** | Swedish, Norwegian, Danish, Finnish |
| **Asia** | Simplified Chinese, Japanese (Planned), Hindi |

Language support is defined via modular **Language Profiles** in `thulium.data.language_profiles`.

## Evaluation & Benchmarking

Thulium includes built-in tools for rigorous evaluation.

```python
from thulium.evaluation.metrics import cer, wer

reference = "The quick brown fox"
hypothesis = "The quick brown fax"

print(f"CER: {cer(reference, hypothesis):.4f}")
print(f"WER: {wer(reference, hypothesis):.4f}")
```

## Contributing

We welcome contributions from the community, especially for adding new language profiles or model architectures. Please refer to [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on code style, testing, and pull requests.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
