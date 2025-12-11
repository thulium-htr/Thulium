# High-Level API

## Functions

### `recognize_image`

```python
def recognize_image(
    path: Union[str, Path],
    language: str = "en",
    pipeline: str = "default",
    device: str = "auto"
) -> PageResult:
```

Recognizes text in a single image file.

### `recognize_pdf`

```python
def recognize_pdf(
    path: Union[str, Path],
    ...
) -> List[PageResult]:
```

Recognizes text in all pages of a PDF.

## Data Structures

### `PageResult`

- `full_text`: `str`
- `lines`: `List[Line]`
- `language`: `str`
