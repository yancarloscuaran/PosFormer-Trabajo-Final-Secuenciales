from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from Pos_Former.datamodule import vocab
from einops import rearrange
from torch import LongTensor
from torchmetrics import Metric


class Hypothesis:
    """
    Clase que almacena una hipótesis generada durante el proceso de Beam Search.
    Contiene la secuencia generada y su puntaje asociado.
    Esta clase es utilizada en el proceso de inferencia para manejar múltiples hipótesis.
    """
    seq: List[int]
    score: float

    def __init__(
        self,
        seq_tensor: LongTensor,
        score: float,
        direction: str,
    ) -> None:
        assert direction in {"l2r", "r2l"}
        raw_seq = seq_tensor.tolist()

        if direction == "r2l":
            result = raw_seq[::-1]  # Invierte la secuencia si es de derecha a izquierda
        else:
            result = raw_seq

        self.seq = result
        self.score = score

    def __len__(self):
        # Retorna la longitud de la secuencia
        if len(self.seq) != 0:
            return len(self.seq)
        else:
            return 1

    def __str__(self):
        # Representación en string de la hipótesis
        return f"seq: {self.seq}, score: {self.score}"


class ExpRateRecorder(Metric):
    """
    Métrica personalizada para calcular la tasa de reconocimiento exacto (ExpRate).
    Compara las secuencias predichas con las secuencias reales y calcula el porcentaje de coincidencias exactas.
    """
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("total_line", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("rec", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, indices_hat: List[List[int]], indices: List[List[int]]):
        # Actualiza los contadores de líneas totales y coincidencias exactas
        for pred, truth in zip(indices_hat, indices):
            pred = vocab.indices2label(pred)
            truth = vocab.indices2label(truth)

            is_same = pred == truth

            if is_same:
                self.rec += 1

            self.total_line += 1

    def compute(self) -> float:
        # Calcula la tasa de reconocimiento exacto
        exp_rate = self.rec / self.total_line
        return exp_rate

def ce_loss(
    output_hat: torch.Tensor,
    output: torch.Tensor,
    ignore_idx: int = vocab.PAD_IDX,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Calcula la pérdida de entropía cruzada (Cross-Entropy Loss).

    Parámetros:
    output_hat: Tensor de predicciones [batch, len, e].
    output: Tensor de etiquetas reales [batch, len].
    ignore_idx: Índice que se debe ignorar en la pérdida (por ejemplo, PAD_IDX).

    Retorna:
    Tensor con el valor de la pérdida.
    """
    flat_hat = rearrange(output_hat, "b l e -> (b l) e")
    flat = rearrange(output, "b l -> (b l)")
    loss = F.cross_entropy(flat_hat, flat, ignore_index=ignore_idx, reduction=reduction)
    return loss 

def ce_loss_all(
    output_hat: torch.Tensor,
    output: torch.Tensor,
    output_hat_layer: torch.Tensor,
    output_layer: torch.Tensor,
    output_hat_pos: torch.Tensor,
    output_pos: torch.Tensor,
    ignore_idx: int = vocab.PAD_IDX,
    reduction: str = "mean",
) -> Tuple[torch.Tensor,torch.Tensor]:
    """
    Calcula la pérdida de entropía cruzada para múltiples salidas (capas y posiciones).

    Parámetros:
    output_hat: Predicciones principales [batch, len, e].
    output: Etiquetas reales principales [batch, len].
    output_hat_layer: Predicciones de capa [batch, len, e].
    output_layer: Etiquetas reales de capa [batch, len].
    output_hat_pos: Predicciones de posición [batch, len, e].
    output_pos: Etiquetas reales de posición [batch, len].

    Retorna:
    Tuple con las pérdidas para las predicciones principales, de capa y de posición.
    """
    flat_hat = rearrange(output_hat, "b l e -> (b l) e")
    flat = rearrange(output, "b l -> (b l)")
    loss = F.cross_entropy(flat_hat, flat, ignore_index=ignore_idx, reduction=reduction)
    flag = flat != ignore_idx

    flat_hat_layer = rearrange(output_hat_layer, "b l e -> (b l) e")
    flat_layer = rearrange(output_layer, "b l -> (b l)")
    loss_layer = F.cross_entropy(flat_hat_layer, flat_layer, reduction='none')
    loss_layer = loss_layer[flag].mean()

    flat_hat_pos = rearrange(output_hat_pos, "b l e -> (b l) e")
    flat_pos = rearrange(output_pos, "b l -> (b l)")
    loss_pos = F.cross_entropy(flat_hat_pos, flat_pos, reduction='none')
    loss_pos = loss_pos[flag].mean()

    return loss , loss_layer , loss_pos

def to_tgt_output(
    tokens: Union[List[List[int]], List[LongTensor]],
    direction: str,
    device: torch.device,
    pad_to_len: Optional[int] = None,
) -> Tuple[LongTensor, LongTensor]:
    """
    Genera tensores de entrada (tgt) y salida (out) para secuencias de tokens.

    Parámetros:
    tokens: Lista de secuencias de índices.
    direction: Dirección de la secuencia ("l2r" o "r2l").
    device: Dispositivo donde se almacenarán los tensores.

    Retorna:
    Tensores tgt y out con las secuencias procesadas.
    """
    assert direction in {"l2r", "r2l"}

    if isinstance(tokens[0], list):
        tokens = [torch.tensor(t, dtype=torch.long) for t in tokens]

    if direction == "l2r":
        tokens = tokens
        start_w = vocab.SOS_IDX
        stop_w = vocab.EOS_IDX
    else:
        tokens = [torch.flip(t, dims=[0]) for t in tokens]
        start_w = vocab.EOS_IDX
        stop_w = vocab.SOS_IDX

    batch_size = len(tokens)
    lens = [len(t) for t in tokens]

    length = max(lens) + 1
    if pad_to_len is not None:
        length = max(length, pad_to_len)

    tgt = torch.full(
        (batch_size, length),
        fill_value=vocab.PAD_IDX,
        dtype=torch.long,
        device=device,
    )
    out = torch.full(
        (batch_size, length),
        fill_value=vocab.PAD_IDX,
        dtype=torch.long,
        device=device,
    )

    for i, token in enumerate(tokens):
        tgt[i, 0] = start_w
        tgt[i, 1 : (1 + lens[i])] = token

        out[i, : lens[i]] = token
        out[i, lens[i]] = stop_w

    return tgt, out

def to_bi_tgt_out(
    tokens: List[List[int]], device: torch.device
) -> Tuple[LongTensor, LongTensor]:
    """
    Genera tensores de entrada y salida bidireccionales para secuencias de tokens.

    Parámetros:
    tokens: Lista de secuencias de índices.
    device: Dispositivo donde se almacenarán los tensores.

    Retorna:
    Tensores tgt y out bidireccionales.
    """
    l2r_tgt, l2r_out = to_tgt_output(tokens, "l2r", device)
    r2l_tgt, r2l_out = to_tgt_output(tokens, "r2l", device)

    tgt = torch.cat((l2r_tgt, r2l_tgt), dim=0)
    out = torch.cat((l2r_out, r2l_out), dim=0)

    return tgt, out
