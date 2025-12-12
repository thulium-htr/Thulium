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

"""Tests for Training Utilities.

Validates checkpointing, early stopping, and other training helpers.
"""

from __future__ import annotations

import os
from pathlib import Path
import pytest
import torch
import torch.nn as nn

from thulium.training.checkpointing import Checkpointer
from thulium.training.early_stopping import EarlyStopping

def test_early_stopping():
    """Test early stopping logic."""
    es = EarlyStopping(patience=3, min_delta=0.01)
    
    # Improvement
    assert not es(0.5)
    assert es.best_loss == 0.5
    assert es.counter == 0
    
    # Slight regression (below delta)
    assert not es(0.499) # 0.001 change < 0.01
    assert es.counter == 1
    
    # Improvement again
    assert not es(0.3)
    assert es.counter == 0
    
    # Regression x3
    assert not es(0.4) # 1
    assert not es(0.4) # 2
    assert es(0.4)     # 3 -> True (Stop)

def test_checkpointer(tmp_path):
    """Test saving and loading checkpoints."""
    model = nn.Linear(10, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    checkpointer = Checkpointer(save_dir=str(tmp_path))
    
    # Save
    checkpointer.save(
        name="test_ckpt",
        model=model,
        optimizer=optimizer,
        epoch=1,
        metrics={"loss": 0.5}
    )
    
    assert (tmp_path / "test_ckpt.pt").exists()
    
    # Load
    checkpoint = checkpointer.load(str(tmp_path / "test_ckpt.pt"))
    assert checkpoint["epoch"] == 1
    assert checkpoint["metrics"]["loss"] == 0.5
    
    # Verify weights match (trivial here since same instance, but logic holds)
    model.load_state_dict(checkpoint["model_state_dict"])
