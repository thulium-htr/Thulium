# Architecture Overview

Thulium is designed as a modular, extensible library for handwritten text recognition (HTR). It separates concerns into clear layers:

## 1. Data Layer (`thulium.data`)
Handles data loading, preprocessing, and abstraction.
- **Loaders**: Ingest images and PDFs using `pdf2image`.
- **Transforms**: Resize, pad, and normalize images for deep learning models.
- **Language Profiles**: Central registry of alphabets and configurations for 50+ languages.

## 2. Model Layer (`thulium.models`)
PyTorch-based implementations of neural network components.
- **Backbones**: Feature update extractors (CNN, ViT).
- **Sequence Heads**: Context modeling (BiLSTM, Transformer).
- **Decoders**: Sequence-to-sequence mapping (CTC, Attention).
- **Wrappers**: High-level classes like `HTRModel` that compose these sub-modules.

## 3. Pipeline Layer (`thulium.pipeline`)
Orchestrates the flow from raw image to structured text.
- **Segmentation**: Breaks pages into lines/words.
- **Recognition**: Runs HTR on segments.
- **Multi-language**: Routes inputs to specific language models.

## 4. API Layer (`thulium.api`)
Exposes functionality to end-users.
- `recognize_image()`: Single function entry point.
- `PageResult`: Structured, typed return object.

## 5. Evaluation & XAI (`thulium.evaluation`, `thulium.xai`)
Tools for researching and improving models.
- **Metrics**: CER, WER standard calculations.
- **XAI**: Visualization of attention maps and confidence scores.
