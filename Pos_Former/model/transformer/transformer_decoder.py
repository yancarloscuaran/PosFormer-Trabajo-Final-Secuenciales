import copy
from functools import partial
from typing import Optional

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .arm import AttentionRefinementModule
from .attention import MultiheadAttention


# Función auxiliar para clonar módulos múltiples veces
# Esto se utiliza para crear múltiples capas del decoder
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class TransformerDecoder(nn.Module):
    """
    Implementa el Transformer Decoder, que incluye múltiples capas de atención y el módulo ARM.
    Este decoder es instanciado desde decoder.py y utiliza las características del encoder
    como memoria para el mecanismo de atención cruzada.
    """
    def __init__(
        self,
        decoder_layer,
        num_layers: int,
        arm: Optional[AttentionRefinementModule],
        norm=None,
    ):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)  # Clonar capas del decoder
        self.num_layers = num_layers
        self.norm = norm

        self.arm = arm  # Módulo ARM para refinar pesos de atención

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        height: int,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_vocab:Optional[Tensor] = None,
    ) -> Tensor:
        """
        Procesa la secuencia objetivo (tgt) a través de las capas del decoder.

        Parámetros:
        tgt: Tensor de entrada al decoder.
        memory: Tensor de memoria proveniente del encoder.
        height: Altura de la memoria para el módulo ARM.
        tgt_mask: Máscara causal para la atención propia.
        memory_mask: Máscara para la atención cruzada.
        tgt_key_padding_mask: Máscara de padding para la secuencia objetivo.
        memory_key_padding_mask: Máscara de padding para la memoria.
        tgt_vocab: Vocabulario objetivo (opcional).

        Retorna:
        Tensor procesado y pesos de atención acumulados.
        """
        output = tgt

        arm = None  # Inicializar el módulo ARM
        for i, mod in enumerate(self.layers):
            # Procesar a través de cada capa del decoder
            output, attn = mod(
                output,
                memory,
                arm,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                tgt_vocab=tgt_vocab
            )
            # Aplicar ARM después de cada capa excepto la última
            if i != len(self.layers) - 1 and self.arm is not None:
                arm = partial(self.arm, attn, memory_key_padding_mask, height)

        if self.norm is not None:
            output = self.norm(output)  # Normalización final

        return output, attn


class TransformerDecoderLayer(nn.Module):
    """
    Implementa una capa del Transformer Decoder, que incluye:
    - Self-Attention con máscara causal para que cada token solo atienda a tokens anteriores.
    - Cross-Attention donde Q proviene del decoder y K, V del encoder.
    - MLP para procesar características.
    - Conexiones residuales y normalización para estabilizar el entrenamiento.
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)  # Self-Attention
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)  # Cross-Attention
        # Implementación del MLP
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Normalización y dropout para estabilizar el entrenamiento
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = F.relu  # Función de activación para el MLP

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = F.relu
        super(TransformerDecoderLayer, self).__setstate__(state)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        arm: Optional[AttentionRefinementModule],
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_vocab:Optional[Tensor] = None,
    ) -> Tensor:
        """
        Procesa la entrada a través de Self-Attention, Cross-Attention y MLP.

        Parámetros:
        tgt: Secuencia objetivo.
        memory: Memoria del encoder.
        arm: Módulo ARM para refinar pesos de atención.
        tgt_mask: Máscara causal para Self-Attention.
        memory_mask: Máscara para Cross-Attention.
        tgt_key_padding_mask: Máscara de padding para la secuencia objetivo.
        memory_key_padding_mask: Máscara de padding para la memoria.
        tgt_vocab: Vocabulario objetivo (opcional).

        Retorna:
        Tensor procesado y pesos de atención.
        """
        # Self-Attention con máscara causal
        tgt2 = self.self_attn(
            tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = tgt + self.dropout1(tgt2)  # Conexión residual
        tgt = self.norm1(tgt)  # Normalización

        # Cross-Attention donde Q proviene del decoder y K, V del encoder
        tgt2, attn = self.multihead_attn(
            tgt,
            memory,
            memory,
            arm=arm,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
            tgt_vocab=tgt_vocab,
        )
        tgt = tgt + self.dropout2(tgt2)  # Conexión residual
        tgt = self.norm2(tgt)  # Normalización

        # MLP para procesar características
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)  # Conexión residual
        tgt = self.norm3(tgt)  # Normalización

        return tgt, attn
