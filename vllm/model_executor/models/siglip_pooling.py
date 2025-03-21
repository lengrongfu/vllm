# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Dict, Any

import torch
from torch import nn
from transformers import SiglipVisionConfig

from vllm.config import VllmConfig
from vllm.model_executor.models.pooling import VllmModelForPooling
from vllm.model_executor.models.siglip import SiglipVisionModel
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.utils import maybe_prefix

class SiglipMultiModalProjector(nn.Module):
    def __init__(self, vision_hidden_size: int, projection_dim: int):
        super().__init__()
        self.linear = nn.Linear(vision_hidden_size, projection_dim, bias=True)

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        return self.linear(image_features.to(self.linear.weight.dtype))

@MULTIMODAL_REGISTRY.register_model()
class SiglipModelForPooling(VllmModelForPooling):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        if not isinstance(vllm_config.model_config.hf_config, SiglipVisionConfig):
            raise ValueError(
                "SiglipModelForPooling requires SiglipVisionConfig, "
                f"got {type(vllm_config.model_config.hf_config).__name__}"
            )
            
        self.vision_tower = SiglipVisionModel(
            vllm_config.model_config.hf_config,
            vllm_config.quant_config,
            prefix=maybe_prefix(prefix, "vision_tower")
        )
        
        self.projector = SiglipMultiModalProjector(
            vision_hidden_size=vllm_config.model_config.hf_config.hidden_size,
            projection_dim=vllm_config.model_config.hf_config.projection_dim
        )

    def pool_features(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        image_embeds: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Pool image features from either raw pixels or precomputed embeddings.
        
        Args:
            pixel_values: Tensor of shape (batch_size, num_channels, height, width)
            image_embeds: Tensor of shape (batch_size, num_features, hidden_size)
            
        Returns:
            Pooled features tensor of shape (batch_size, hidden_size)
        """
        if image_embeds is not None:
            if image_embeds.ndim != 3:
                raise ValueError(
                    f"image_embeds must be 3D tensor, got {image_embeds.ndim}D"
                )
            return image_embeds.mean(dim=1)
            
        if pixel_values is None:
            raise ValueError("Must provide either pixel_values or image_embeds")
            
        if pixel_values.ndim != 4:
            raise ValueError(
                f"pixel_values must be 4D tensor, got {pixel_values.ndim}D"
            )
            
        vision_output = self.vision_tower(pixel_values)
        projected = self.projector(vision_output)
        return projected.mean(dim=1)