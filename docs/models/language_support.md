# Language Support

Thulium provides comprehensive multilingual support through a centralized language profile system. Each language profile encapsulates character sets, tokenization strategies, and decoder configurations.

## Supported Languages Overview

Thulium currently supports **52 languages** across **12 distinct writing systems**.

| Script Family | Languages | Direction |
| :--- | :--- | :--- |
| Latin | 30+ languages | LTR |
| Cyrillic | Russian, Ukrainian, Bulgarian, Serbian | LTR |
| Arabic | Arabic, Persian, Urdu | RTL |
| Hebrew | Hebrew | RTL |
| Devanagari | Hindi, Marathi | LTR |
| Bengali | Bengali | LTR |
| Tamil | Tamil | LTR |
| Telugu | Telugu | LTR |
| Georgian | Georgian | LTR |
| Armenian | Armenian | LTR |
| Thai | Thai | LTR |
| CJK | Chinese, Japanese, Korean | LTR/TTB |

---

## Scandinavian Languages

The Scandinavian language group uses extended Latin alphabets with distinctive characters.

| Code | Language | Special Characters | Notes |
| :--- | :--- | :--- | :--- |
| `nb` | Norwegian (Bokmal) | ae, o-slash, a-ring | Standard written Norwegian |
| `nn` | Norwegian (Nynorsk) | ae, o-slash, a-ring | New Norwegian variant |
| `sv` | Swedish | a-umlaut, o-umlaut, a-ring | |
| `da` | Danish | ae, o-slash, a-ring | |
| `is` | Icelandic | eth, thorn, acute accents | Preserves Old Norse characters |
| `fo` | Faroese | eth, acute accents | |
| `fi` | Finnish | a-umlaut, o-umlaut | Finno-Ugric; not Indo-European |

---

## Baltic Languages

| Code | Language | Special Characters | Notes |
| :--- | :--- | :--- | :--- |
| `lt` | Lithuanian | ogonek, caron, macron | |
| `lv` | Latvian | macron, cedilla, caron | |
| `et` | Estonian | a-umlaut, o-tilde, o-umlaut | Finno-Ugric language |

---

## Caucasus Region

| Code | Language | Script | Special Features |
| :--- | :--- | :--- | :--- |
| `az` | Azerbaijani | Latin | Schwa, soft-g, dotless-i, cedilla |
| `tr` | Turkish | Latin | Dotted I / dotless i distinction |
| `ka` | Georgian | Georgian | 33-letter Mkhedruli script |
| `hy` | Armenian | Armenian | 39-letter alphabet |

### Azerbaijani Character Set

The Azerbaijani alphabet includes 32 letters with the following distinctive characters:

- Schwa (uppercase/lowercase)
- Soft-g (breve)
- Dotless-i (uppercase/lowercase)
- O-umlaut, U-umlaut
- C-cedilla, S-cedilla

---

## Western European Languages

| Code | Language | Special Characters | Notes |
| :--- | :--- | :--- | :--- |
| `en` | English | None (ASCII base) | Baseline model |
| `de` | German | a-umlaut, o-umlaut, u-umlaut, eszett | |
| `fr` | French | Acute, grave, circumflex, cedilla, ligatures | |
| `es` | Spanish | Acute accents, n-tilde, inverted punctuation | |
| `pt` | Portuguese | Acute, circumflex, tilde, cedilla | |
| `it` | Italian | Acute, grave, circumflex | |
| `nl` | Dutch | Acute, umlaut, circumflex | |

---

## Eastern European Languages

### Latin Script

| Code | Language | Special Characters | Notes |
| :--- | :--- | :--- | :--- |
| `pl` | Polish | Ogonek, acute, kreska, dot | |
| `cs` | Czech | Caron, acute | |
| `sk` | Slovak | Caron, acute, circumflex | |
| `hu` | Hungarian | Acute, double-acute, umlaut | |
| `ro` | Romanian | Breve, comma-below, circumflex | |
| `hr` | Croatian | Caron, acute, stroke | |
| `sl` | Slovenian | Caron | |

### Cyrillic Script

| Code | Language | Alphabet Size | Notes |
| :--- | :--- | :--- | :--- |
| `ru` | Russian | 33 letters | Standard Cyrillic |
| `uk` | Ukrainian | 33 letters | Includes i, yi, ye |
| `bg` | Bulgarian | 30 letters | No hard sign |
| `sr` | Serbian (Cyrillic) | 30 letters | Also supports Latin |

### Greek Script

| Code | Language | Alphabet Size | Notes |
| :--- | :--- | :--- | :--- |
| `el` | Greek | 24 letters | Plus polytonic variants |

---

## Middle East and Central Asia

Right-to-left (RTL) script support includes proper handling of text direction and character shaping.

| Code | Language | Script | Direction | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `ar` | Arabic | Arabic | RTL | Includes Arabic-Indic numerals |
| `fa` | Persian (Farsi) | Arabic | RTL | Extended Arabic with pe, che, zhe, gaf |
| `ur` | Urdu | Arabic | RTL | Extended Arabic alphabet |
| `he` | Hebrew | Hebrew | RTL | 22 consonant letters |

---

## South Asian Languages

### Devanagari Script

| Code | Language | Base Script | Notes |
| :--- | :--- | :--- | :--- |
| `hi` | Hindi | Devanagari | 47 primary characters |
| `mr` | Marathi | Devanagari | Shared character set with Hindi |

### Other Indic Scripts

| Code | Language | Script | Notes |
| :--- | :--- | :--- | :--- |
| `bn` | Bengali | Bengali | 11 vowels, 39 consonants |
| `ta` | Tamil | Tamil | 12 vowels, 18 consonants |
| `te` | Telugu | Telugu | 16 vowels, 36 consonants |
| `gu` | Gujarati | Gujarati | Placeholder |
| `pa` | Punjabi | Gurmukhi | Placeholder |
| `kn` | Kannada | Kannada | Placeholder |
| `ml` | Malayalam | Malayalam | Placeholder |

---

## East Asian Languages

| Code | Language | Script | Notes |
| :--- | :--- | :--- | :--- |
| `zh` | Chinese (Simplified) | Han | Common character subset |
| `ja` | Japanese | Mixed | Hiragana + Katakana; Kanji requires extended vocabulary |
| `ko` | Korean | Hangul | Jamo-based representation |

---

## Southeast Asian Languages

| Code | Language | Script | Notes |
| :--- | :--- | :--- | :--- |
| `th` | Thai | Thai | 44 consonants, 15 vowel symbols |
| `vi` | Vietnamese | Latin | Extensive diacritic system (tones) |
| `id` | Indonesian | Latin | Standard Latin |
| `ms` | Malay | Latin | Standard Latin |

---

## African Languages

| Code | Language | Script | Notes |
| :--- | :--- | :--- | :--- |
| `sw` | Swahili | Latin | Standard Latin |
| `af` | Afrikaans | Latin | Dutch-derived diacritics |

---

## API Usage

### Retrieving Language Profiles

```python
from thulium.data.language_profiles import get_language_profile, list_supported_languages

# Get specific profile
profile = get_language_profile("az")
print(f"Language: {profile.name}")
print(f"Script: {profile.script}")
print(f"Alphabet size: {len(profile.alphabet)}")

# List all supported languages
languages = list_supported_languages()
print(f"Supported: {len(languages)} languages")
```

### Filtering by Region or Script

```python
from thulium.data.language_profiles import get_languages_by_region, get_languages_by_script

# Get Scandinavian languages
nordic = get_languages_by_region("Scandinavia")

# Get all Cyrillic languages
cyrillic = get_languages_by_script("Cyrillic")
```

---

## Adding New Languages

To add support for a new language:

1. Define the character set (alphabet)
2. Create a `LanguageProfile` entry in `language_profiles.py`
3. Optionally, create a language-specific pipeline configuration
4. Run validation: `validate_language_profile(profile)`

See [Contributing Guide](../../CONTRIBUTING.md) for detailed instructions.
