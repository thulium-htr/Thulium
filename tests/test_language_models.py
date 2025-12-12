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

"""Tests for Language Models.

Tests N-gram and Neural Language Models functionality.
"""

from __future__ import annotations

import pytest
import torch
from thulium.models.language_models.ngram_lm import CharacterNGramLM
from thulium.models.language_models.word_lm import TransformerWordLM

def test_ngram_lm_training_and_score():
    """Test N-gram LM training and scoring."""
    texts = ["hello", "hello world", "world"]
    lm = CharacterNGramLM(n=3, smoothing='add_k', add_k=1.0)
    lm.train(texts)
    
    # "hel" should be probable
    score_known = lm.score([ord(c) for c in "hel"])
    # "xyz" should be less probable (only unigram/smoothed prob)
    score_unknown = lm.score([ord(c) for c in "xyz"])
    
    assert score_known > score_unknown
    assert score_known < 0.0

def test_ngram_lm_partial_scoring():
    """Test incremental scoring matches full scoring."""
    lm = CharacterNGramLM(n=3)
    lm.train(["abcde"])
    
    tokens = [ord(c) for c in "abc"]
    
    # Full score
    full_score = lm.score(tokens)
    
    # Incremental score
    # Score 'a' given start
    log_p1, ctx1 = lm.score_partial([], tokens[0])
    # Score 'b' given 'a'
    log_p2, ctx2 = lm.score_partial([tokens[0]], tokens[1], state=ctx1)
    # Score 'c' given 'ab'
    log_p3, ctx3 = lm.score_partial([tokens[0], tokens[1]], tokens[2], state=ctx2)
    
    # Note: score() sums log probs of P(c1|start) + P(c2|c1) + ...
    # Our incremental score_partial might differ slightly depending on BOS handling implementation detail,
    # but generally sum(partials) should approx full_score.
    
    # In our implementation: score(tokens) -> P(a|BOS) * P(b|a) * P(c|b)
    total_partial = log_p1 + log_p2 + log_p3
    
    assert abs(total_partial - full_score) < 1e-5

def test_transformer_word_lm_shapes():
    """Test Neural Word LM forward pass dimensions."""
    B, T, V = 2, 10, 100
    model = TransformerWordLM(vocab_size=V, d_model=32, nhead=2, num_layers=1)
    
    x = torch.randint(0, V, (B, T))
    logits = model(x)
    
    assert logits.shape == (B, T, V)

def test_transformer_word_lm_scoring():
    """Test scoring wrapper."""
    model = TransformerWordLM(vocab_size=50)
    model.eval()
    
    seq = [1, 5, 10, 20]
    score = model.score(seq)
    
    assert isinstance(score, float)
    assert score < 0 # Log probs are negative
