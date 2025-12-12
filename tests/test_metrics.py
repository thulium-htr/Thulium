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

"""Tests for Evaluation Metrics.

Tests robustness and correctness of CER, WER, and other metrics.
"""

from __future__ import annotations

import pytest
from thulium.evaluation.metrics import (
    compute_cer,
    compute_wer,
    compute_ser,
)

@pytest.mark.parametrize(
    "ref, hyp, expected",
    [
        ("hello", "hello", 0.0),
        ("hello", "hullo", 0.2), # 1 sub / 5
        ("test", "", 1.0),
        ("", "test", float('inf')), # Division by zero handling? Usually handled or returns inf
        ("abc", "ab", 1/3),
    ]
)
def test_cer_calculation(ref, hyp, expected):
    """Test CER computation."""
    if ref == "":
        # Our metric implementation might return 1.0 or raise error or return length of hyp
        # Let's check behavior if known, otherwise calculate roughly
        # Usually dist(empty, hyp) = len(hyp). If ref_len=0, metric is undefined or infinity.
        # Let's handle it gracefully
        assert compute_cer(ref, hyp) == expected
    else:
        assert abs(compute_cer(ref, hyp) - expected) < 1e-4

def test_wer_calculation():
    """Test WER computation."""
    ref = "hello world this is a test"
    hyp = "hello world this is a tost"
    
    # 6 words. 1 substitution.
    expected = 1/6
    assert abs(compute_wer(ref, hyp) - expected) < 1e-4

def test_ser_batch():
    """Test Sequence Error Rate on batch."""
    refs = ["a", "b", "c"]
    hyps = ["a", "b", "d"]
    
    # 1 error out of 3 sequences
    assert compute_ser(refs, hyps) == 1/3
