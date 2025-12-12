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

"""Tests for HTR Decoders.

Tests CTC decoder functionality including greedy and beam search decoding.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from thulium.models.decoders.ctc_decoder import (
    BeamSearchConfig,
    CTCDecoder,
)


@pytest.fixture
def decoder(request):
    """Create a decoder instance."""
    return CTCDecoder(input_size=128, num_classes=30, blank_index=0)

def test_decoder_initialization(decoder):
    """Test standard initialization."""
    assert decoder.blank_index == 0
    assert decoder.num_classes == 30
    assert isinstance(decoder.fc, torch.nn.Linear)

def test_decoder_forward(decoder):
    """Test forward pass dimensions."""
    B, T, D = 4, 50, 256  # Input dimensions usually come from Head
    # The decoder expects (B, T, input_size) or (T, B, input_size) depending on implementation
    # Based on our ctc_decoder.py, it likely takes (B, T, D)
    
    # We need to match input_size of decoder
    decoder = CTCDecoder(input_size=256, num_classes=10, blank_index=0)
    x = torch.randn(4, 50, 256)
    
    logits = decoder(x)
    assert logits.shape == (4, 50, 11)  # num_classes + blank

def test_greedy_decoding():
    """Test greedy decoding logic."""
    decoder = CTCDecoder(input_size=16, num_classes=4, blank_index=0)
    
    # Manually construct log probs to force specific output
    # Classes: 0(blank), 1, 2, 3, 4
    # Sequence: 1, 0, 2, 2, 0 -> Output: 1, 2, 2
    
    T, B, C = 5, 1, 5
    log_probs = torch.full((T, B, C), -100.0) # Init low prob
    
    # Set high probs for desired path
    # Time 0: Class 1
    log_probs[0, 0, 1] = 0.0
    # Time 1: Blank
    log_probs[1, 0, 0] = 0.0
    # Time 2: Class 2
    log_probs[2, 0, 2] = 0.0
    # Time 3: Class 2 (repeat) -> Should be merged in standard CTC? 
    # Wait, consecutive duplicates are merged. 
    # To get "2, 2", we need "2, blank, 2" or distinct time steps
    # If we output "2, 2" directly, greedy decoder usually merges them to "2"
    log_probs[3, 0, 2] = 0.0
    # Time 4: Blank
    log_probs[4, 0, 0] = 0.0
    
    # Expected: [1, 2] because 2,2 merges to 2
    expected = [[1, 2]]
    
    # Note: Our decoder expects (B, T, C) input usually for forward, 
    # but decode_greedy might accept (T, B, C) or (B, T, C).
    # Let's check implementation. Usually log_probs are (T, B, C) for CTCLoss.
    # Our implementation in Step 724 viewed earlier:
    # "predictions = torch.argmax(log_probs, dim=2)" -> implies dim 2 is classes.
    # If log_probs is (B, T, C), then dim=2 is classes.
    # Let's assume input to decode_greedy matches standard PyTorch convention if not specified.
    
    # Let's verify shape assumption from previous code view or assume (B, T, C).
    log_probs_batch = log_probs.transpose(0, 1) # (B, T, C)
    
    result = decoder.decode_greedy(log_probs_batch)
    assert result == expected

def test_beam_search_decoding():
    """Test beam search runs without error."""
    decoder = CTCDecoder(input_size=32, num_classes=10, blank_index=0)
    
    B, T, C = 2, 10, 11
    log_probs = torch.randn(B, T, C) # (B, T, C)
    log_probs = torch.nn.functional.log_softmax(log_probs, dim=-1)
    
    config = BeamSearchConfig(beam_width=5)
    
    # Should handle simple inputs
    result = decoder.decode_beam_search(log_probs, config=config)
    
    assert len(result) == B
    assert isinstance(result[0], list)
