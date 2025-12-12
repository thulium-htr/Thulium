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

"""Smoke tests for Thulium package.

This module contains basic smoke tests to ensure the package structure,
versioning, and top-level imports are functioning correctly. These tests
should pass for any valid installation of the Thulium package.

Test Categories:
    - Version verification
    - Top-level import validation
    - Deprecation warning checks

Example:
    Running these tests:

    $ pytest tests/test_basic.py -v
"""

from __future__ import annotations

import warnings

import pytest

import thulium
from thulium import __version__


class TestVersioning:
    """Tests for package versioning."""

    def test_version_exists(self) -> None:
        """Verify that the package version is exposed.

        The version should be a non-empty string with semantic versioning
        format (at least X.Y.Z).
        """
        assert __version__ is not None
        assert isinstance(__version__, str)
        assert len(__version__.split(".")) >= 3

    def test_version_format(self) -> None:
        """Verify version follows semantic versioning format.

        Version should be in format X.Y.Z where X, Y, Z are integers.
        """
        parts = __version__.split(".")
        for part in parts[:3]:
            assert part.isdigit() or part.split("-")[0].isdigit()


class TestImports:
    """Tests for package imports."""

    def test_top_level_imports(self) -> None:
        """Verify top-level modules can be imported without error.

        All major submodules should be importable from the top-level
        thulium package.
        """
        try:
            from thulium import api
            from thulium import cli
            from thulium import data
            from thulium import evaluation
            from thulium import models
            from thulium import pipeline
            from thulium import training
        except ImportError as e:
            pytest.fail(f"Failed to import top-level module: {e}")

    def test_no_deprecation_warnings(self) -> None:
        """Ensure no deprecation warnings are emitted during import.

        Package imports should not trigger deprecation warnings from
        Thulium code (third-party library warnings are excluded).
        """
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            import thulium.api

            # Filter to only warnings from thulium code
            relevant_warnings = [
                x
                for x in w
                if "thulium" in str(x.filename)
                and "FutureWarning" not in str(x.category)
            ]
            assert (
                len(relevant_warnings) == 0
            ), f"Deprecation warnings found: {relevant_warnings}"
