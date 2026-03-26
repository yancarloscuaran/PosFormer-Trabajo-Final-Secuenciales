import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch import Tensor
from torch.nn.modules.batchnorm import BatchNorm1d


class MaskBatchNorm2d(nn.Module):
    """
    Implementa una capa de normalización por lotes (BatchNorm) adaptada para trabajar con máscaras.
    """
    def __init__(self, num_features: int):
        super().__init__()
        self.bn = BatchNorm1d(num_features)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """
        Aplica la normalización por lotes solo en las regiones no enmascaradas del tensor de entrada.

        Parámetros:
        x: Tensor de entrada con dimensiones [b, d, h, w].
        mask: Máscara binaria con dimensiones [b, 1, h, w].

        Retorna:
        Tensor normalizado con las mismas dimensiones que la entrada.
        """
        x = rearrange(x, "b d h w -> b h w d")
        mask = mask.squeeze(1)

        not_mask = ~mask

        # Normaliza únicamente las regiones no enmascaradas.
        flat_x = x[not_mask, :]
        flat_x = self.bn(flat_x)
        x[not_mask, :] = flat_x

        x = rearrange(x, "b h w d -> b d h w")

        return x


class AttentionRefinementModule(nn.Module):
    """
    Implementa el módulo de refinamiento de atención (ARM), también conocido como IAC (Implicit Attention Correction).

    Este módulo ajusta las probabilidades de atención antes de la normalización Softmax, corrigiendo el término de cobertura
    únicamente para los símbolos entidad, mientras deja los símbolos estructurales con peso cero.
    """
    def __init__(self, nhead: int, dc: int, cross_coverage: bool, self_coverage: bool):
        super().__init__()
        assert cross_coverage or self_coverage
        self.nhead = nhead
        self.cross_coverage = cross_coverage
        self.self_coverage = self_coverage

        # Determina el número de canales de entrada basado en las configuraciones de cobertura cruzada y propia.
        if cross_coverage and self_coverage:
            in_chs = 2 * nhead
        else:
            in_chs = nhead

        # Convolución para procesar las atenciones acumuladas.
        self.conv = nn.Conv2d(in_chs, dc, kernel_size=5, padding=2)
        self.act = nn.ReLU(inplace=True)

        # Proyección final para ajustar las dimensiones al número de cabezas de atención.
        self.proj = nn.Conv2d(dc, nhead, kernel_size=1, bias=False)
        self.post_norm = MaskBatchNorm2d(nhead)

    def forward(
        self, prev_attn: Tensor, key_padding_mask: Tensor, h: int, curr_attn: Tensor, tgt_vocab: Tensor
    ) -> Tensor:
        """
        Realiza el refinamiento de atención acumulada.

        Este método es llamado desde transformer_decoder.py en cada capa del decoder.

        Parámetros:
        prev_attn: Atenciones acumuladas de pasos anteriores [(b * nhead), t, l].
        key_padding_mask: Máscara de padding para las claves [b, l].
        h: Altura de la imagen procesada.
        curr_attn: Atenciones actuales [(b * nhead), t, l].
        tgt_vocab: Tokens objetivo para identificar símbolos estructurales y entidades [b, l].

        Retorna:
        Tensor refinado con las atenciones ajustadas [(b * nhead), t, l].
        """
        t = curr_attn.shape[1]
        # Expande la máscara de padding para que coincida con las dimensiones de las atenciones.
        mask = repeat(key_padding_mask, "b (h w) -> (b t) () h w", h=h, t=t)

        # Reorganiza las dimensiones de las atenciones para procesarlas por lotes.
        curr_attn = rearrange(curr_attn, "(b n) t l -> b n t l", n=self.nhead)
        prev_attn = rearrange(prev_attn, "(b n) t l -> b n t l", n=self.nhead)
        b = curr_attn.shape[0] // 2

        # Acumula las atenciones cruzadas y propias según la configuración.
        attns = []
        if self.cross_coverage:
            attns.append(prev_attn)
        if self.self_coverage:
            attns.append(curr_attn)
        attns = torch.cat(attns, dim=1)

        # Identifica los símbolos estructurales y ajusta sus pesos a cero.
        tgt_vocab = tgt_vocab.unsqueeze(1).repeat(1, 2 * self.nhead, 1)
        mask_vocab = torch.logical_not(torch.logical_or(tgt_vocab == 110, torch.logical_or(tgt_vocab == 82, tgt_vocab == 83)))
        attns = attns * mask_vocab.unsqueeze(-1).float()

        # Calcula las atenciones acumuladas y resta el término de cobertura.
        attns = attns.cumsum(dim=2) - attns
        attns = rearrange(attns, "b n t (h w) -> (b t) n h w", h=h)

        # Aplica convoluciones y activaciones para refinar las atenciones.
        cov = self.conv(attns)
        cov = self.act(cov)

        # Enmascara las regiones no válidas.
        cov = cov.masked_fill(mask, 0.0)
        cov = self.proj(cov)

        # Normaliza las atenciones refinadas.
        cov = self.post_norm(cov, mask)

        # Reorganiza las dimensiones para devolver el tensor refinado.
        cov = rearrange(cov, "(b t) n h w -> (b n) t (h w)", t=t)
        return cov


