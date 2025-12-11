import torch.nn as nn
import torch

class AttentionDecoder(nn.Module):
    """
    Autoregressive Attention-based decoder.
    Used for sequence-to-sequence decoding where alignment is learned implicitly.
    """
    def __init__(self, hidden_size=256, vocab_size=100):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, vocab_size)

    def forward(self, encoder_out, hidden=None):
        """
        Stub forward pass.
        In reality, this would take target inputs for training (forcing)
        or loop for inference.
        """
        # (B, Seq, Hidden)
        batch_size = encoder_out.size(0)
        # Dummy output for compatibility
        return torch.randn(batch_size, 10, self.out.out_features)
