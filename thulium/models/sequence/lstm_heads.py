import torch
import torch.nn as nn

class BiLSTMHead(nn.Module):
    """
    Bidirectional LSTM head for sequence modeling.
    Maps 2D feature maps (B, C, H, W) -> Sequence (B, W, HiddenSize).
    """
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 2):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) features from backbone.
        Returns:
            (B, W, HiddenSize)
        """
        b, c, h, w = x.size()
        # Collapse H dimension: average pooling or flattening.
        # Here we assume H=1 after backbone or we average.
        x = x.mean(dim=2) # (B, C, W)
        x = x.permute(0, 2, 1) # (B, W, C)
        
        output, _ = self.rnn(x)
        output = self.proj(output)
        return output
