import torch.nn as nn
import torch

class TransformerHead(nn.Module):
    """
    Transformer Encoder as a sequence head.
    Replaces BiLSTMs for clearer long-range dependency modeling.
    """
    def __init__(self, input_size=256, hidden_size=256, nhead=4, num_layers=4):
        super().__init__()
        self.input_proj = nn.Linear(input_size, hidden_size)
        layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x):
        """
        Args:
            x: (B, Seq, InputSize)
        """
        x = self.input_proj(x)
        return self.encoder(x)
