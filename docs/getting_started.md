# Getting Started

## Prerequisites

- Python 3.10+
- PyTorch 2.0+
- `poppler-utils` (for PDF processing)

## Installation

```bash
git clone https://github.com/olaflaitinen/Thulium.git
cd Thulium
pip install -e .[dev]
```

## Basic Usage

### CLI

```bash
thulium recognize data/sample.png -l en
```

### Python

```python
from thulium.api import recognize_image

res = recognize_image("data/sample.png", language="en")
print(res.full_text)
```
