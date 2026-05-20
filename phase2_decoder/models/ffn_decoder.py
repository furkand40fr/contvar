"""
FFNDecoder

Takes the encoder embedding → returns N_go_terms logits.

forward() does NOT apply sigmoid:
    - During training, BCEWithLogitsLoss applies sigmoid internally (numerical stability).
    - During inference, torch.sigmoid() is called explicitly.
"""

import torch
import torch.nn as nn


class FFNDecoder(nn.Module):

    def __init__(self, input_dim: int, n_classes: int,
                 hidden_dims: list[int] = None, dropout: float = 0.3):
        """
        input_dim   : encoder_output_dim (256)
        n_classes   : GO vocab size (including NULL_FUNCTION)
        hidden_dims : hidden layer sizes, default [512, 1024, 512]
        """
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [512, 1024, 512]

        layers = []
        in_dim = input_dim
        for i, out_dim in enumerate(hidden_dims):
            # Slightly lower dropout on the last hidden layer
            p = dropout if i < len(hidden_dims) - 1 else dropout * 0.67
            layers += [
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(),
                nn.Dropout(p),
            ]
            in_dim = out_dim

        layers.append(nn.Linear(in_dim, n_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x      : [B, input_dim]
        return : [B, n_classes]  — logits (sigmoid not applied)
        """
        return self.net(x)
