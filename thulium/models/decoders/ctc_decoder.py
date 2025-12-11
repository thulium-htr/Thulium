import torch
import torch.nn as nn
import torch.nn.functional as F

class CTCDecoder(nn.Module):
    """
    CTC Decoder layer.
    """
    def __init__(self, input_size: int, num_classes: int):
        super().__init__()
        # +1 for blank token
        self.fc = nn.Linear(input_size, num_classes + 1)
        self.blank_index = 0  # Typically 0 is blank in PyTorch CTC

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, W, HiddenSize)
        Returns:
            Log probabilities (B, W, NumClasses + 1)
        """
        x = self.fc(x)
        return F.log_softmax(x, dim=2)
    
    def decode_greedy(self, log_probs: torch.Tensor, blank_idx: int = 0) -> list:
        """
        Simple greedy decode.
        """
        predictions = torch.argmax(log_probs, dim=2)
        batch_results = []
        for batch_idx in range(predictions.size(0)):
            seq = predictions[batch_idx].cpu().numpy()
            decoded_tokens = []
            prev = None
            for token in seq:
                if token != prev and token != blank_idx:
                    decoded_tokens.append(token)
                prev = token
            batch_results.append(decoded_tokens)
        return batch_results
