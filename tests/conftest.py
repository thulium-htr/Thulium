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

"""Pytest fixtures and configuration for Thulium test suite.

This module defines common fixtures and configuration used across all Thulium
tests. It provides consistent test data, temporary file handling, and shared
model instances to speed up testing.

Fixtures:
    temp_dir: Session-scoped temporary directory for test artifacts.
    sample_image: Function-scoped dummy PIL Image for testing.
    sample_batch: Function-scoped dummy tensor batch for model testing.
    device: Session-scoped device string ('cpu' or 'cuda').

Example:
    Using fixtures in tests:

    >>> def test_image_processing(sample_image):
    ...     result = process_image(sample_image)
    ...     assert result is not None
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Generator

import pytest
import torch
from PIL import Image


@pytest.fixture(scope="session")
def temp_dir(tmp_path_factory: pytest.TempPathFactory) -> Generator[Path, None, None]:
    """Create a session-scoped temporary directory.

    This directory persists across all tests in the session and is
    cleaned up automatically when the test session ends.

    Args:
        tmp_path_factory: Pytest's temporary path factory.

    Yields:
        Path to the temporary directory.
    """
    temp_path = tmp_path_factory.mktemp("thulium_tests")
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture(scope="function")
def sample_image() -> Image.Image:
    """Create a dummy white image for testing.

    Creates a 256x64 white RGB image suitable for testing
    image processing functions.

    Returns:
        PIL Image object.
    """
    return Image.new("RGB", (256, 64), color="white")


@pytest.fixture(scope="function")
def sample_batch() -> torch.Tensor:
    """Create a dummy batch of images for model testing.

    Creates a batch of 4 random images with shape (B, C, H, W)
    where B=4, C=3, H=64, W=256.

    Returns:
        Tensor of shape (4, 3, 64, 256).
    """
    return torch.randn(4, 3, 64, 256)


@pytest.fixture(scope="session")
def device() -> str:
    """Return the device to use for testing.

    Automatically selects CUDA if available, otherwise CPU.

    Returns:
        Device string: 'cuda' or 'cpu'.
    """
    return "cuda" if torch.cuda.is_available() else "cpu"
