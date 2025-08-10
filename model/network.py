import torch
import torch.nn as nn
from typing import Dict, Any

from data.preset import PresetIndexesHelper
from model.layer import Transformer
from model.position_encoding import PositionEmbeddingSine, PositionalEncoding1D


class CNNBackbone(nn.Module):
    """
    CNN feature extractor for the backbone of Transformer encoder
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            self._conv_block(in_channels, out_channels // 16, 5,  batch_norm=False),
            self._conv_block(out_channels // 16, out_channels // 8, 4),
            self._conv_block(out_channels // 8, out_channels // 4, 4),
            self._conv_block(out_channels // 4, out_channels // 2, 4),
            self._conv_block(out_channels // 2, out_channels, 4),
        )

    def _conv_block(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 2,
        padding: int = 2,
        batch_norm: bool = True,
    ) -> nn.Sequential:
        """
        Create a conv -> activation -> (BN) block.
        """
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
        layers.append(nn.LeakyReLU(0.1))

        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))

        return nn.Sequential(*layers)

    def forward(self, melspec: torch.Tensor) -> torch.Tensor:
        return self.conv(melspec)
    

class SynthRL(nn.Module):
    """
    SynthTR consists of CNN backbone, Transformer encoder and Transformer decoder.
    Output of the Transformer encoder serves as keys and values of the decoder.
    The transformer decoder receives learnable queries for synthesizer parameters.
    """

    def __init__(
            self,
            preset_idx_helper: PresetIndexesHelper,
            d_model: int = 256,
            in_channels: int = 1,
            n_queries: int = 144,
            transformer_kwargs: Dict[str, Any] = {},
        ):
        super().__init__()
        self.out_dim = preset_idx_helper._learnable_preset_size
        self.cat_idx, self.num_idx = self._get_learnable_idx(preset_idx_helper)

        self.backbone = CNNBackbone(in_channels, d_model)
        self.enc_pos_embed = PositionEmbeddingSine(d_model // 2)
        self.query_pos_embed = PositionalEncoding1D(d_model, n_queries)
        self.transformer = Transformer(n_queries, d_model, **transformer_kwargs)

        # Projection heads
        self.proj_dropout = nn.Dropout(0.3)
        self.proj = nn.Linear(d_model, self.out_dim)
        self.last_act = nn.Tanh()
        proj = []

        for i in range(len(self.cat_idx)):
            proj.append(nn.Linear(d_model, len(self.cat_idx[i])))

        for i in range(len(self.num_idx)):
            proj.append(nn.Linear(d_model, 1))
        
        self.proj = nn.ModuleList(proj)

    def forward(self, spectrogram):
        features = self.backbone(spectrogram)
        enc_pos_embed = self.enc_pos_embed(features)
        query_pos_embed = self.query_pos_embed(features)

        # Transformer encoder-decoder
        dec_out = self.transformer(features, query_pos_embed, enc_pos_embed)

        # Projection heads
        batch_size, n_query, d_model = dec_out.shape
        out = torch.zeros((batch_size, self.out_dim), device=dec_out.device)
        
        dec_out = dec_out.reshape(-1, d_model)
        dec_out = self.proj_dropout(dec_out)
        dec_out = dec_out.reshape(batch_size, n_query, -1)

        for i in range(len(self.cat_idx)):
            out[:, self.cat_idx[i]] = self.proj[i](dec_out[:, i, :])

        for i in range(len(self.cat_idx), n_query):
            out[:, self.num_idx[i - len(self.cat_idx)]] = self.proj[i](dec_out[:, i, :]).squeeze()

        # Output Activation
        out = self.last_act(out)
        out = 0.5 * (out + 1.)
        return out
    
    def _get_learnable_idx(self, preset_idx_helper):
        full_idx = preset_idx_helper.full_to_learnable
        cat_idx, num_idx = [], []

        for idx in full_idx:
            if isinstance(idx, list):
                cat_idx.append(idx)
            elif isinstance(idx, int):
                num_idx.append(idx)

        return cat_idx, num_idx
