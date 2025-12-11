# Low-Level API Reference

This document describes the internal APIs for advanced users who need direct access to model components, pipeline configuration, and language profiles.

## Pipeline Layer

### `HTRPipeline`

The core pipeline class that orchestrates recognition.

```python
from thulium.pipeline.htr_pipeline import HTRPipeline
from thulium.pipeline.config import load_pipeline_config

config = load_pipeline_config("htr_en_default")
pipeline = HTRPipeline(config, device="cuda", language="en")

result = pipeline.process("document.png", language="en")
```

**Constructor Parameters:**

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `config` | `dict` | Required | Pipeline configuration dictionary |
| `device` | `str` | `"auto"` | Computation device |
| `language` | `str` | `None` | Default language code |

**Methods:**

| Method | Returns | Description |
| :--- | :--- | :--- |
| `process(image_path, language)` | `PageResult` | Run full recognition pipeline |

---

### `load_pipeline_config`

Load pipeline configuration from YAML files.

```python
from thulium.pipeline.config import load_pipeline_config

# Load built-in configuration
config = load_pipeline_config("htr_az_default")

# Load custom configuration
config = load_pipeline_config("path/to/custom.yaml")
```

---

## Language Profiles

### `LanguageProfile`

Dataclass containing language-specific configuration.

```python
from thulium.data.language_profiles import LanguageProfile

@dataclass
class LanguageProfile:
    code: str                           # ISO 639-1 code
    name: str                           # Human-readable name
    script: str                         # Writing system
    alphabet: List[str]                 # Character set
    direction: str = "LTR"              # Text direction
    region: str = "Global"              # Geographic region
    special_tokens: List[str]           # Reserved tokens
    tokenizer_type: str = "char"        # Tokenization strategy
    default_decoder: str = "ctc_beam"   # Default decoder
    default_language_model: str = None  # Optional LM
    notes: str = ""                     # Additional notes
```

**Methods:**

| Method | Returns | Description |
| :--- | :--- | :--- |
| `get_vocab_size()` | `int` | Total vocabulary size including special tokens |
| `get_char_to_idx()` | `Dict[str, int]` | Character to index mapping |
| `get_idx_to_char()` | `Dict[int, str]` | Index to character mapping |

---

### Profile Access Functions

```python
from thulium.data.language_profiles import (
    get_language_profile,
    list_supported_languages,
    get_languages_by_region,
    get_languages_by_script,
    validate_language_profile,
)

# Get specific profile
profile = get_language_profile("az")

# List all languages
languages = list_supported_languages()

# Filter by region
scandinavian = get_languages_by_region("Scandinavia")

# Filter by script
cyrillic_langs = get_languages_by_script("Cyrillic")

# Validate a profile
validate_language_profile(profile)  # Raises ValueError if invalid
```

---

## Model Components

### Backbones

Feature extraction networks.

```python
from thulium.models.backbones.cnn_backbone import CNNBackbone
from thulium.models.backbones.vit_backbone import ViTBackbone

# CNN backbone
cnn = CNNBackbone(pretrained=True)

# Vision Transformer backbone
vit = ViTBackbone(image_size=224, patch_size=16)
```

---

### Sequence Heads

Temporal modeling layers.

```python
from thulium.models.sequence.bilstm_head import BiLSTMHead
from thulium.models.sequence.transformer_head import TransformerHead

# BiLSTM
lstm = BiLSTMHead(input_size=512, hidden_size=256, num_layers=2)

# Transformer
transformer = TransformerHead(d_model=512, nhead=8, num_layers=6)
```

---

### Decoders

Output sequence generation.

```python
from thulium.models.decoders.ctc_decoder import CTCDecoder
from thulium.models.decoders.attention_decoder import AttentionDecoder

# CTC decoder
ctc = CTCDecoder(input_size=512, num_classes=100)
output = ctc(features)
decoded = ctc.decode_greedy(output)

# Attention decoder
attention = AttentionDecoder(hidden_size=512, vocab_size=100)
```

---

### Language Models

Optional language model scoring.

```python
from thulium.models.language_models.char_lm import CharacterLanguageModel
from thulium.models.language_models.word_lm import WordLanguageModel

# Character-level LM
char_lm = CharacterLanguageModel(vocab_size=100, hidden_size=256)

# Word-level LM
word_lm = WordLanguageModel(vocab_size=50000, embedding_dim=300)
```

---

## Evaluation

### Metrics Functions

```python
from thulium.evaluation.metrics import cer, wer, ser

reference = "The quick brown fox"
hypothesis = "The quich brown fax"

print(f"CER: {cer(reference, hypothesis):.4f}")
print(f"WER: {wer(reference, hypothesis):.4f}")
print(f"SER: {ser(reference, hypothesis):.4f}")
```

---

## XAI (Explainability)

### Attention Visualization

```python
from thulium.xai.attention_maps import visualize_attention

# Visualize model attention on input
visualize_attention(model, image, output_path="attention.png")
```

### Confidence Analysis

```python
from thulium.xai.confidence_analysis import analyze_confidence

# Analyze per-character confidence
analysis = analyze_confidence(result)
low_confidence = [c for c in analysis if c.confidence < 0.5]
```

### Error Analysis

```python
from thulium.xai.error_analysis import categorize_errors

# Categorize recognition errors
errors = categorize_errors(reference, hypothesis)
for error in errors:
    print(f"{error.type}: '{error.expected}' -> '{error.actual}'")
```
