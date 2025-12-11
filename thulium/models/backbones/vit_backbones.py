import torch.nn as nn
import torch

class ViTBackbone(nn.Module):
    """
    Vision Transformer (ViT) backbone for feature extraction.
    Specialized for variable width line images using patch embedding.
    """
    def __init__(self):
        super().__init__()
        # Stub implementation representing a patch embedding layer and encoder
        self.patch_embed = nn.Conv2d(3, 768, kernel_size=16, stride=16)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=8)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=6)

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            features: (B, Hidden, H', W') or (B, Seq, Hidden)
        """
        x = self.patch_embed(x) # (B, 768, H/16, W/16)
        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2) # (B, Seq, 768)
        x = self.transformer(x)
        # Reshape back to feature map if needed, or return sequence
        return x
