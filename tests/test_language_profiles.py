# Copyright 2025 Thulium Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for Language Profiles.

Validates the integrity and configuration of supported language profiles.
Ensures over 50+ languages are correctly defined with alphabets and scripts.
"""

from __future__ import annotations

import pytest
from thulium.data.language_profiles import (
    get_language_profile,
    list_supported_languages,
    get_languages_by_script,
    LanguageProfile,
    UnsupportedLanguageError,
)

def test_english_profile():
    """Verify default English profile."""
    profile = get_language_profile("en")
    assert profile.name == "English"
    assert profile.script == "Latin"
    assert "a" in profile.alphabet
    assert "A" in profile.alphabet
    assert profile.direction == "LTR"

def test_multilingual_coverage():
    """Ensure we support a wide range of languages."""
    langs = list_supported_languages()
    assert len(langs) >= 50, f"Expected 50+ languages, found {len(langs)}"
    
    # Check key regions
    assert "az" in langs # Azerbaijani
    assert "tr" in langs # Turkish
    assert "ru" in langs # Russian
    assert "ar" in langs # Arabic
    assert "is" in langs # Icelandic (Scandinavia)

def test_script_filtering():
    """Test filtering by script."""
    cyrillic = get_languages_by_script("Cyrillic")
    assert "ru" in cyrillic
    assert "uk" in cyrillic
    assert "en" not in cyrillic

def test_alphabet_consistency():
    """Check alphabet integrity for specific complex profiles."""
    # Azerbaijani
    az = get_language_profile("az")
    assert "ə" in az.alphabet
    assert "Ə" in az.alphabet # Case handling might differ
    
    # German
    de = get_language_profile("de")
    assert "ß" in de.alphabet
    assert "ä" in de.alphabet

def test_invalid_language():
    """Test error handling for unknown languages."""
    with pytest.raises(UnsupportedLanguageError):
        get_language_profile("invalid_code_123")
