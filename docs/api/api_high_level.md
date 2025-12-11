# High-Level API Reference

This document describes the public API functions available in `thulium.api` for text recognition tasks.

## Functions

### `recognize_image`

Recognize handwritten text in a single image file.

```python
def recognize_image(
    path: Union[str, Path],
    language: str = "en",
    pipeline: str = "default",
    device: str = "auto"
) -> PageResult:
```

**Parameters:**

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `path` | `str` or `Path` | Required | Path to the input image file (PNG, JPEG, etc.) |
| `language` | `str` | `"en"` | ISO 639-1 language code |
| `pipeline` | `str` | `"default"` | Pipeline configuration name |
| `device` | `str` | `"auto"` | Computation device: `"cpu"`, `"cuda"`, or `"auto"` |

**Returns:**

`PageResult` object containing recognized text and metadata.

**Raises:**

| Exception | Condition |
| :--- | :--- |
| `FileNotFoundError` | Image file does not exist |
| `UnsupportedLanguageError` | Language code is not in the registry |

**Example:**

```python
from thulium.api import recognize_image

result = recognize_image(
    path="document.png",
    language="az",
    device="cuda"
)

print(result.full_text)
for line in result.lines:
    print(f"[{line.confidence:.2f}] {line.text}")
```

---

### `recognize_pdf`

Recognize handwritten text in all pages of a PDF document.

```python
def recognize_pdf(
    path: Union[str, Path],
    language: str = "en",
    pipeline: str = "default",
    device: str = "auto"
) -> List[PageResult]:
```

**Parameters:**

Same as `recognize_image`.

**Returns:**

List of `PageResult` objects, one per page.

**Requirements:**

Requires `poppler-utils` installed for PDF rendering.

**Example:**

```python
from thulium.api import recognize_pdf

results = recognize_pdf("multi_page.pdf", language="de")
for i, page in enumerate(results):
    print(f"--- Page {i+1} ---")
    print(page.full_text)
```

---

### `recognize_batch`

Process multiple image files in batch.

```python
def recognize_batch(
    paths: List[Union[str, Path]],
    language: str = "en",
    pipeline: str = "default",
    device: str = "auto"
) -> List[PageResult]:
```

**Parameters:**

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `paths` | `List[str]` or `List[Path]` | Required | List of image file paths |
| `language` | `str` | `"en"` | ISO 639-1 language code |
| `pipeline` | `str` | `"default"` | Pipeline configuration name |
| `device` | `str` | `"auto"` | Computation device |

**Returns:**

List of `PageResult` objects, one per input image.

**Example:**

```python
from pathlib import Path
from thulium.api import recognize_batch

images = list(Path("documents/").glob("*.png"))
results = recognize_batch(images, language="tr")
```

---

## Data Structures

### `PageResult`

Container for recognition results from a single page or image.

```python
@dataclass
class PageResult:
    full_text: str
    lines: List[Line]
    language: str
    metadata: Optional[Dict[str, Any]] = None
```

**Attributes:**

| Attribute | Type | Description |
| :--- | :--- | :--- |
| `full_text` | `str` | Complete recognized text, lines joined by newlines |
| `lines` | `List[Line]` | Individual line results with positions |
| `language` | `str` | Language code used for recognition |
| `metadata` | `Dict` | Additional information (device, model, etc.) |

**Methods:**

| Method | Returns | Description |
| :--- | :--- | :--- |
| `to_dict()` | `Dict` | Serialize to dictionary for JSON export |

---

### `Line`

Individual text line with spatial and confidence information.

```python
@dataclass
class Line:
    text: str
    confidence: float
    bbox: Tuple[int, int, int, int]
```

**Attributes:**

| Attribute | Type | Description |
| :--- | :--- | :--- |
| `text` | `str` | Recognized text content |
| `confidence` | `float` | Model confidence score (0.0 to 1.0) |
| `bbox` | `Tuple[int, int, int, int]` | Bounding box (x, y, width, height) |

---

## Error Handling

### `UnsupportedLanguageError`

Raised when a requested language code is not available.

```python
from thulium.data.language_profiles import UnsupportedLanguageError

try:
    result = recognize_image("test.png", language="xyz")
except UnsupportedLanguageError as e:
    print(f"Language not supported: {e.language_code}")
    print(f"Available: {e.available_languages[:10]}...")
```

---

## Pipeline Caching

The API caches pipeline instances to avoid reloading models on repeated calls. Pipelines are keyed by `(pipeline_name, device)` tuple.

To clear the cache:

```python
from thulium.api.recognize import _PIPELINE_CACHE
_PIPELINE_CACHE.clear()
```
