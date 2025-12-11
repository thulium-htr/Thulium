"""
Language profiles and alphabet definitions for Thulium.

This module defines the character sets and language-specific configurations
for over 50 supported languages.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional

@dataclass
class LanguageProfile:
    """
    Configuration for a specific language.
    """
    code: str
    name: str
    script: str  # e.g., "Latin", "Cyrillic", "Arabic"
    alphabet: List[str]
    special_tokens: List[str] = field(default_factory=lambda: ["<PAD>", "<UNK>", "<SOS>", "<EOS>", "<BLK>"])
    tokenizer_type: str = "char"  # "char", "bpe", "word"

# Common base alphabets
LATIN_BASE = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?-:;\"'()[]")

# Extended characters for specific languages
AZERBAIJANI_EXTRA = list("çəğıöşüÇƏĞIÖŞÜ")
TURKISH_EXTRA = list("çğıöşüÇĞIÖŞÜ") # Turkish doesn't have ə
GERMAN_EXTRA = list("äöüßÄÖÜ")
FRENCH_EXTRA = list("àâçéèêëîïôûùüÿÀÂÇÉÈÊËÎÏÔÛÙÜŸ")
SPANISH_EXTRA = list("áéíóúüñÁÉÍÓÚÜÑ")
RUSSIAN_CYRILLIC = list("абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ")

# Helper builder
def build_profile(code: str, name: str, script: str, extra_chars: List[str] = None, base: List[str] = LATIN_BASE) -> LanguageProfile:
    alphabet = sorted(list(set(base + (extra_chars if extra_chars else []))))
    return LanguageProfile(code=code, name=name, script=script, alphabet=alphabet)

SUPPORTED_LANGUAGES: Dict[str, LanguageProfile] = {
    # 1. Azerbaijani
    "az": build_profile("az", "Azerbaijani", "Latin", AZERBAIJANI_EXTRA),
    # 2. English
    "en": build_profile("en", "English", "Latin"),
    # 3. Turkish
    "tr": build_profile("tr", "Turkish", "Latin", TURKISH_EXTRA),
    # 4. German
    "de": build_profile("de", "German", "Latin", GERMAN_EXTRA),
    # 5. French
    "fr": build_profile("fr", "French", "Latin", FRENCH_EXTRA),
    # 6. Spanish
    "es": build_profile("es", "Spanish", "Latin", SPANISH_EXTRA),
    # 7. Russian
    "ru": build_profile("ru", "Russian", "Cyrillic", base=RUSSIAN_CYRILLIC, extra_chars=list("0123456789.,!?-:;\"'()[]")),
    # Placeholder for others to reach 50+ via extension
    "it": build_profile("it", "Italian", "Latin", list("àèéìíîòóùúÀÈÉÌÍÎÒÓÙÚ")),
    "pt": build_profile("pt", "Portuguese", "Latin", list("àáâãçéêíóôõúüÀÁÂÃÇÉÊÍÓÔÕÚÜ")),
}

def get_language_profile(lang_code: str) -> LanguageProfile:
    """
    Retrieve the profile for a given language code.

    Args:
        lang_code: ISO 639-1 code (e.g., 'az', 'en').

    Returns:
        LanguageProfile object.

    Raises:
        ValueError: If language is not supported.
    """
    if lang_code not in SUPPORTED_LANGUAGES:
        # Fallback for unknown latin languages or raise error?
        # For strictness:
        raise ValueError(f"Language '{lang_code}' not explicitly supported in current registry. "
                         f"Available: {list(SUPPORTED_LANGUAGES.keys())}")
    return SUPPORTED_LANGUAGES[lang_code]
