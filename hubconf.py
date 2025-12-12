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

"""PyTorch Hub configuration for Thulium models.

This module enables loading Thulium models via PyTorch Hub:

    import torch
    model = torch.hub.load('olaflaitinen/Thulium', 'thulium_base')

See https://pytorch.org/hub/ for more information.
"""

from __future__ import annotations

dependencies = ["torch", "torchvision", "PIL"]


def thulium_tiny(pretrained: bool = True, **kwargs):
    """
    Thulium Tiny model for edge/mobile deployment.
    
    Args:
        pretrained: Load pretrained weights.
        **kwargs: Additional model configuration.
        
    Returns:
        HTRPipeline: Configured HTR pipeline.
        
    Example:
        >>> import torch
        >>> model = torch.hub.load('olaflaitinen/Thulium', 'thulium_tiny')
    """
    from thulium import HTRPipeline
    
    if pretrained:
        return HTRPipeline.from_pretrained("thulium-tiny", **kwargs)
    return HTRPipeline.from_config("config/models/htr_cnn_lstm_ctc_tiny.yaml", **kwargs)


def thulium_base(pretrained: bool = True, **kwargs):
    """
    Thulium Base model - balanced accuracy and speed.
    
    Args:
        pretrained: Load pretrained weights.
        **kwargs: Additional model configuration.
        
    Returns:
        HTRPipeline: Configured HTR pipeline.
        
    Example:
        >>> import torch
        >>> model = torch.hub.load('olaflaitinen/Thulium', 'thulium_base')
    """
    from thulium import HTRPipeline
    
    if pretrained:
        return HTRPipeline.from_pretrained("thulium-base", **kwargs)
    return HTRPipeline.from_config("config/models/htr_cnn_lstm_ctc_base.yaml", **kwargs)


def thulium_large(pretrained: bool = True, **kwargs):
    """
    Thulium Large model for maximum accuracy.
    
    Args:
        pretrained: Load pretrained weights.
        **kwargs: Additional model configuration.
        
    Returns:
        HTRPipeline: Configured HTR pipeline.
        
    Example:
        >>> import torch
        >>> model = torch.hub.load('olaflaitinen/Thulium', 'thulium_large')
    """
    from thulium import HTRPipeline
    
    if pretrained:
        return HTRPipeline.from_pretrained("thulium-large", **kwargs)
    return HTRPipeline.from_config("config/models/htr_vit_transformer_seq2seq_large.yaml", **kwargs)


def thulium_multilingual(pretrained: bool = True, **kwargs):
    """
    Thulium Multilingual model supporting 52+ languages.
    
    Args:
        pretrained: Load pretrained weights.
        **kwargs: Additional model configuration.
        
    Returns:
        HTRPipeline: Configured HTR pipeline.
        
    Example:
        >>> import torch
        >>> model = torch.hub.load('olaflaitinen/Thulium', 'thulium_multilingual')
        >>> result = model.recognize(image, language="de")
    """
    from thulium import HTRPipeline
    
    if pretrained:
        return HTRPipeline.from_pretrained("thulium-multilingual", **kwargs)
    return HTRPipeline.from_config("config/models/htr_vit_transformer_seq2seq_large.yaml", **kwargs)
