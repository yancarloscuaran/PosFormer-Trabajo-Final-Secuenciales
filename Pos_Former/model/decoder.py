from typing import List
from typing import Tuple
import torch
import torch.nn as nn
from einops import rearrange
from torch import FloatTensor, LongTensor
import numpy as np
from Pos_Former.datamodule import vocab, vocab_size 
from Pos_Former.model.pos_enc import WordPosEnc
from Pos_Former.model.transformer.arm import AttentionRefinementModule
from Pos_Former.model.transformer.transformer_decoder import (
    TransformerDecoder,
    TransformerDecoderLayer,
)
from Pos_Former.utils.generation_utils import DecodeModel, PosDecodeModel

# Función auxiliar para construir el TransformerDecoder con o sin el módulo de refinamiento de atención (ARM).
def _build_transformer_decoder(
    d_model: int,
    nhead: int,
    num_decoder_layers: int,
    dim_feedforward: int,
    dropout: float,
    dc: int,
    cross_coverage: bool,
    self_coverage: bool,
) -> nn.TransformerDecoder:
    # Crea una capa básica de TransformerDecoder.
    decoder_layer = TransformerDecoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
    )
    # Si se especifica, añade el módulo de refinamiento de atención (ARM).
    if cross_coverage or self_coverage:
        arm = AttentionRefinementModule(nhead, dc, cross_coverage, self_coverage)
    else:
        arm = None

    # Construye el TransformerDecoder con las capas y el ARM (si aplica).
    decoder = TransformerDecoder(decoder_layer, num_decoder_layers, arm)
    return decoder

class Decoder(DecodeModel):
    """
    Implementa el Word Decoder, encargado de generar la secuencia de tokens LaTeX a partir de las características extraídas por el encoder.
    """
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        dc: int,
        cross_coverage: bool,
        self_coverage: bool,
    ):
        super().__init__()

        # Embedding para los tokens del vocabulario LaTeX, seguido de una normalización.
        self.word_embed = nn.Sequential(
            nn.Embedding(vocab_size, d_model), nn.LayerNorm(d_model)
        )
        # Positional Encoding para añadir información de posición a los embeddings de los tokens.
        self.pos_enc = WordPosEnc(d_model=d_model)

        # Normalización final antes de pasar al TransformerDecoder.
        self.norm = nn.LayerNorm(d_model)

        # Construcción del TransformerDecoder principal.
        self.model = _build_transformer_decoder(
            d_model=d_model,
            nhead=nhead,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            dc=dc,
            cross_coverage=cross_coverage,
            self_coverage=self_coverage,
        )

        # Proyección final para mapear las salidas del Transformer al vocabulario LaTeX.
        self.proj = nn.Linear(d_model, vocab_size)

    def _build_attention_mask(self, length):
        # Construye una máscara causal triangular para garantizar que el modelo no vea tokens futuros.
        mask = torch.full(
            (length, length), fill_value=1, dtype=torch.bool, device=self.device
        )
        mask.triu_(1)  # Zeros en la diagonal inferior.
        return mask

    def forward(
        self, src: FloatTensor, src_mask: LongTensor, tgt: LongTensor 
    ) -> FloatTensor:
        """
        Realiza el paso hacia adelante del Word Decoder.

        Este método es llamado desde posformer.py para generar las predicciones de tokens LaTeX.

        Parámetros:
        src: Características extraídas por el encoder [b, h, w, d].
        src_mask: Máscara de las características [b, h, w].
        tgt: Secuencia objetivo de tokens LaTeX [b, l].

        Retorna:
        Salidas proyectadas sobre el vocabulario LaTeX [b, l, vocab_size].
        """
        _, l = tgt.size()
        # Máscara causal para evitar que el modelo vea tokens futuros.
        tgt_mask = self._build_attention_mask(l)
        tgt_pad_mask = tgt == vocab.PAD_IDX

        tgt_vocab=tgt
        # Embedding de los tokens objetivo.
        tgt = self.word_embed(tgt)  # [b, l, d]
        tgt = self.pos_enc(tgt)  # Añade información posicional.
        tgt = self.norm(tgt)
        
        # Reorganización de las dimensiones para el TransformerDecoder.
        h = src.shape[1]
        src = rearrange(src, "b h w d -> (h w) b d")
        src_mask = rearrange(src_mask, "b h w -> b (h w)")
        tgt = rearrange(tgt, "b l d -> l b d")

        # Paso por el TransformerDecoder.
        out, attn  = self.model(
            tgt=tgt,
            memory=src,
            height=h,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_mask,
            tgt_vocab=tgt_vocab,
        )
    
        # Proyección final sobre el vocabulario LaTeX.
        out_rearrange = rearrange(out, "l b d -> b l d")
        out = self.proj(out_rearrange)
        return out, attn

    def transform(
        self, src: List[FloatTensor], src_mask: List[LongTensor], input_ids: LongTensor
    ) -> FloatTensor:
        """
        Método utilizado durante la inferencia para transformar las características del encoder en secuencias de salida.

        Este método es llamado desde posformer.py en el contexto de inferencia.
        """
        assert len(src) == 1 and len(src_mask) == 1
        word_out, _ = self(src[0], src_mask[0], input_ids)
        return word_out

class PosDecoder(PosDecodeModel):
    """
    Implementa el Pos Decoder, encargado de calcular las posiciones estructurales y de capa para los tokens generados.
    """
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        dc: int,
        cross_coverage: bool,
        self_coverage: bool,
    ):
        super().__init__()
        # Embedding para las posiciones objetivo, utilizando una proyección lineal en lugar de nn.Embedding.
        self.pos_embed = nn.Sequential(
            nn.Linear(5,d_model),nn.GELU(),nn.LayerNorm(d_model)
        )  # Proyección de [2b,l,5] a [2b,l,256].
        self.pos_enc = WordPosEnc(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.norm = nn.LayerNorm(d_model)

        # Reutilización del TransformerDecoder para procesar las posiciones.
        self.model = _build_transformer_decoder(
            d_model=d_model,
            nhead=nhead,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            dc=dc,
            cross_coverage=cross_coverage,
            self_coverage=self_coverage,
        )
        # Proyección para calcular las posiciones estructurales.
        self.layernum_proj = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 5)
        ) 
        # Proyección para calcular las posiciones de capa.
        self.pos_proj = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 6)
        ) 
    def _build_attention_mask(self, length):
        # Construye una máscara causal triangular para garantizar que el modelo no vea tokens futuros.
        mask = torch.full(
            (length, length), fill_value=1, dtype=torch.bool, device=self.device
        )
        mask.triu_(1)  # Zeros en la diagonal inferior.    

        return mask

    def forward(
        self, src: FloatTensor, src_mask: LongTensor, tgt: LongTensor,pos_tgt:FloatTensor
    ) -> Tuple[ FloatTensor,FloatTensor]:
        """
        Realiza el paso hacia adelante del Pos Decoder.

        Este método es llamado desde posformer.py para calcular las posiciones estructurales y de capa.

        Parámetros:
        src: Características extraídas por el encoder [b, h, w, d].
        src_mask: Máscara de las características [b, h, w].
        tgt: Secuencia objetivo de tokens LaTeX [b, l].
        pos_tgt: Embedding de las posiciones objetivo [b, l, 5].

        Retorna:
        Salidas proyectadas para las posiciones estructurales y de capa.
        """

        b , l = tgt.size()
        # Máscara causal para evitar que el modelo vea tokens futuros.
        tgt_mask = self._build_attention_mask(l)
        tgt_pad_mask = tgt == vocab.PAD_IDX
        tgt_vocab=tgt
        # Embedding de las posiciones objetivo.
        pos_tgt=self.pos_embed(pos_tgt)  # Proyección inicial.
        pos_tgt = self.pos_enc(pos_tgt)  # Añade información posicional.
        pos_tgt = self.norm(pos_tgt)

        # Reorganización de las dimensiones para el TransformerDecoder.
        h = src.shape[1]
        src = rearrange(src, "b h w d -> (h w) b d")
        src_mask = rearrange(src_mask, "b h w -> b (h w)")
        pos_tgt = rearrange(pos_tgt, "b l d -> l b d")

        # Paso por el TransformerDecoder.
        out, attn = self.model(
            tgt=pos_tgt,
            memory=src,
            height=h,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_mask,
            tgt_vocab=tgt_vocab,
        )
        out_rearrange = rearrange(out, "l b d -> b l d")
        # Proyecciones finales para las posiciones estructurales y de capa.
        out_pos=self.pos_proj(out_rearrange)
        out_layernum=self.layernum_proj(out_rearrange)
        return out_layernum , out_pos, attn

    def transform(
        self, src: List[FloatTensor], src_mask: List[LongTensor], input_ids: LongTensor
    ) -> FloatTensor:
        """
        Método utilizado durante la inferencia para transformar las características del encoder en posiciones de salida.

        Este método es llamado desde posformer.py en el contexto de inferencia.
        """
        assert len(src) == 1 and len(src_mask) == 1
        out_pos, _ = self(src[0], src_mask[0], input_ids,torch.zeros(1, dtype=torch.float, device=self.device))
        return out_pos
