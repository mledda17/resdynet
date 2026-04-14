from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class ResDyNetConfig:
    n_u: int
    n_y: int
    n_x: int
    n_a: int
    n_b: int
    m: int                  # maximum output lag included in decoder window (0 <= m <= n_a)
    horizon: int
    encoder_hidden: List[int]
    transition_hidden: int
    transition_blocks: int
    decoder_hidden: List[int]
    activation: str = "relu"
    use_bias_A: bool = True
    use_bias_B: bool = False
    use_layer_norm: bool = False

    @property
    def n_gamma(self) -> int:
        return (self.m + 1) * self.n_y
