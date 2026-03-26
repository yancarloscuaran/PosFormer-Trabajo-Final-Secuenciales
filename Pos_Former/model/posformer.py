from typing import List ,Tuple
import numpy as np
import pytorch_lightning as pl
import torch
from torch import FloatTensor, LongTensor
import os
from Pos_Former.utils.utils import Hypothesis

from .decoder import Decoder , PosDecoder
from .encoder import Encoder
from Pos_Former.datamodule import vocab , label_make_muti

class PosFormer(pl.LightningModule):
    """
    Clase principal de la arquitectura PosFormer.
    Integra el encoder, el Word Decoder y el Pos Decoder para realizar el reconocimiento de expresiones matemáticas manuscritas.
    """
    def __init__(
        self,
        d_model: int,
        growth_rate: int,
        num_layers: int,
        nhead: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        dc: int,
        cross_coverage: bool,
        self_coverage: bool,
    ):
        """
        Inicializa los componentes principales del modelo PosFormer.

        - Encoder: Extrae características de alto nivel de la imagen de entrada.
        - Word Decoder: Genera la secuencia de palabras (tokens) a partir de las características del encoder.
        - Pos Decoder: Calcula las posiciones estructurales de los tokens en la expresión matemática.
        """
        super().__init__()
        self.encoder = Encoder(
            d_model=d_model, growth_rate=growth_rate, num_layers=num_layers
        )
        self.decoder = Decoder(
            d_model=d_model,
            nhead=nhead,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            dc=dc,
            cross_coverage=cross_coverage,
            self_coverage=self_coverage,
        )
        self.posdecoder = PosDecoder(
            d_model=d_model,
            nhead=nhead,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            dc=dc,
            cross_coverage=cross_coverage,
            self_coverage=self_coverage
        )
        self.save_path = 'attn_PosFormer'

    def forward(
        self, img: FloatTensor, img_mask: LongTensor, tgt: LongTensor, logger
    ) -> Tuple[FloatTensor,FloatTensor,FloatTensor]:
        """
        Realiza el paso hacia adelante del modelo para procesar una imagen y generar las predicciones.

        Parámetros:
        - img: Tensor de entrada que representa la imagen [b, 1, h, w].
        - img_mask: Máscara de la imagen para indicar las áreas válidas [b, h, w].
        - tgt: Secuencia objetivo en formato bidireccional [2b, l].

        Retorna:
        - out: Predicciones del Word Decoder [2b, l, vocab_size].
        - out_layernum: Salidas del Pos Decoder relacionadas con las capas [2b, l, 5].
        - out_pos: Salidas del Pos Decoder relacionadas con las posiciones [2b, l, 6].
        """
        # El encoder extrae características de la imagen y genera un mapa de características y una máscara asociada.
        feature, mask = self.encoder(img, img_mask)  # [b, t, d]

        # Duplica el batch para manejar simultáneamente las direcciones izquierda a derecha (L2R) y derecha a izquierda (R2L).
        feature = torch.cat((feature, feature), dim=0)  # [2b, t, d]
        mask = torch.cat((mask, mask), dim=0)
        
        # Convierte las secuencias objetivo en etiquetas múltiples para el Pos Decoder.
        tgt_list=tgt.cpu().numpy().tolist()
        muti_labels=label_make_muti.tgt2muti_label(tgt_list)
        muti_labels_tensor=torch.FloatTensor(muti_labels)   #[2b,l,5]
        muti_labels_tensor=muti_labels_tensor.cuda()
        
        # El Word Decoder genera las predicciones de los tokens de la expresión matemática.
        out, _ = self.decoder(feature, mask, tgt)

        # El Pos Decoder genera las posiciones estructurales y de capa para los tokens.
        out_layernum , out_pos, _ =self.posdecoder(feature, mask,tgt,muti_labels_tensor)

        return out, out_layernum, out_pos   # [2b,l,vocab_size], [2b,l,5] and[2b,l,6]

    def beam_search(
        self,
        img: FloatTensor,
        img_mask: LongTensor,
        beam_size: int,
        max_len: int,
        alpha: float,
        early_stopping: bool,
        temperature: float,
        **kwargs,
    ) -> List[Hypothesis]:
        """
        Realiza la búsqueda con haz bidireccional (L2R y R2L) para inferencia.

        Parámetros:
        - img: Tensor de entrada que representa la imagen [b, 1, h', w'].
        - img_mask: Máscara de la imagen para indicar las áreas válidas [b, h', w'].
        - beam_size: Tamaño del haz para la búsqueda.
        - max_len: Longitud máxima de la secuencia generada.

        Retorna:
        - seq_out: Lista de hipótesis generadas por el Word Decoder.

        Nota: Durante la inferencia, solo se utiliza el Word Decoder, ya que el Pos Decoder no participa en esta etapa.
        """
        # El encoder extrae características de la imagen para la inferencia.
        feature, mask = self.encoder(img, img_mask)  # [b, t, d]

        # El Word Decoder realiza la búsqueda con haz para generar la secuencia de salida.
        seq_out= self.decoder.beam_search(
            [feature], [mask], beam_size, max_len, alpha, early_stopping, temperature
        )

        return seq_out