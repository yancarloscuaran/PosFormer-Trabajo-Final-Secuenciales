import math
from typing import Optional

import pytorch_lightning as pl
import torch
from einops import rearrange, repeat


class WordPosEnc(pl.LightningModule):
    """
    Implementa un Positional Encoding sinusoidal fijo para secuencias de tokens.
    Este módulo se utiliza tanto en el Word Decoder como en el Pos Decoder.

    El parámetro `temperature` controla la frecuencia de las ondas sinusoidales,
    permitiendo que diferentes dimensiones del embedding tengan diferentes escalas.
    """
    def __init__(
        self, d_model: int = 512, max_len: int = 500, temperature: float = 10000.0
    ) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len, dtype=torch.float)
        dim_t = torch.arange(0, d_model, 2, dtype=torch.float)
        div_term = 1.0 / (temperature ** (dim_t / d_model))

        # Calcular las ondas sinusoidales para cada posición y dimensión
        inv_freq = torch.einsum("i, j -> i j", position, div_term)

        pe[:, 0::2] = inv_freq.sin()
        pe[:, 1::2] = inv_freq.cos()
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Añade el encoding posicional a las características de entrada.

        Parámetros
        ----------
        x : torch.Tensor
            [b, l, d]

        Retorna
        -------
        torch.Tensor
            [b, l, d]
        """
        _, seq_len, _ = x.size()
        emb = self.pe[:seq_len, :]
        x = x + emb[None, :, :]
        return x


class ImgPosEnc(pl.LightningModule):
    """
    Implementa un Positional Encoding 2D para imágenes.
    Este módulo se utiliza en el encoder para añadir información posicional a las características visuales.

    Combina las posiciones en los ejes x e y para generar un vector de posición único
    para cada región de la imagen.
    """
    def __init__(
        self,
        d_model: int = 512,
        temperature: float = 10000.0,
        normalize: bool = False,
        scale: Optional[float] = None,
    ):
        super().__init__()
        assert d_model % 2 == 0
        self.half_d_model = d_model // 2
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x: torch.Tensor, mask: torch.LongTensor) -> torch.Tensor:
        """
        Añade el encoding posicional 2D a las características de entrada.

        Parámetros
        ----------
        x : torch.Tensor
            [b, h, w, d]
        mask: torch.LongTensor
            [b, h, w]

        Retorna
        -------
        torch.Tensor
            [b, h, w, d]
        """
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(
            0, self.half_d_model, 2, dtype=torch.float, device=self.device
        )
        inv_feq = 1.0 / (self.temperature ** (dim_t / self.half_d_model))

        # Calcular las posiciones sinusoidales para x e y
        pos_x = torch.einsum("b h w, d -> b h w d", x_embed, inv_feq)
        pos_y = torch.einsum("b h w, d -> b h w d", y_embed, inv_feq)

        pos_x = torch.stack((pos_x.sin(), pos_x.cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y.sin(), pos_y.cos()), dim=4).flatten(3)
        pos = torch.cat((pos_x, pos_y), dim=3)

        x = x + pos
        return x


def rotate_every_two(x: torch.FloatTensor):
    # Función auxiliar para rotar pares de dimensiones en embeddings rotatorios
    x = rearrange(x, "... (d j) -> ... d j", j=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d j -> ... (d j)")


class WordRotaryEmbed(pl.LightningModule):
    """
    Implementa un Rotary Positional Embedding para secuencias de tokens.
    Este método es una alternativa al encoding sinusoidal estándar y permite
    rotar las posiciones en pares de dimensiones.
    """
    def __init__(self, d_model: int = 512, temperature: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (
            temperature ** (torch.arange(0, d_model, 2).float() / d_model)
        )
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x: torch.FloatTensor):
        """
        Aplica el encoding rotatorio a las características de entrada.

        Parámetros
        ----------
        x : torch.Tensor
            [b, l, d]

        Retorna
        -------
        torch.Tensor
            [b, l, d]
        """
        _, n, _ = x.size()
        t = torch.arange(n, device=self.device).type_as(self.inv_freq)
        sinusoid_inp = torch.einsum("i, j -> i j", t, self.inv_freq)
        sin, cos = sinusoid_inp.sin(), sinusoid_inp.cos()

        sin, cos = map(lambda t: repeat(t, "b n -> b (n j)", j=2), (sin, cos))

        x = (x * cos) + (rotate_every_two(x) * sin)
        return x


class ImageRotaryEmbed(pl.LightningModule):
    """
    Implementa un Rotary Positional Embedding 2D para imágenes.
    Este método es una generalización del WordRotaryEmbed para trabajar con datos 2D.
    """
    def __init__(
        self,
        d_model: int = 512,
        temperature: float = 10000,
        normalize: bool = False,
        scale: Optional[float] = None,
    ):
        super().__init__()
        assert d_model % 2 == 0
        self.half_d_model = d_model // 2
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x: torch.Tensor, mask: torch.LongTensor) -> torch.Tensor:
        """
        Aplica el encoding rotatorio 2D a las características de entrada.

        Parámetros
        ----------
        x : torch.Tensor
            [b, h, w, d]
        mask: torch.LongTensor
            [b, h, w]

        Retorna
        -------
        torch.Tensor
            [b, h, w, d]
        """
        not_mask = ~mask
        embed_y = not_mask.cumsum(1, dtype=torch.float32)
        embed_x = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            embed_y = embed_y / (embed_y[:, -1:, :] + eps) * self.scale
            embed_x = embed_x / (embed_x[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(
            0, self.half_d_model, 2, dtype=torch.float, device=self.device
        )
        inv_feq = 1.0 / (self.temperature ** (dim_t / self.half_d_model))

        # Calcular posiciones sinusoidales para x e y
        pos_x = torch.einsum("b h w, d -> b h w d", embed_x, inv_feq)
        pos_y = torch.einsum("b h w, d -> b h w d", embed_y, inv_feq)

        # Combinar posiciones sinusoidales
        sin_x, cos_x, sin_y, cos_y = map(
            lambda t: repeat(t, "b h w d -> b h w (d n)", n=2),
            (pos_x.sin(), pos_x.cos(), pos_y.sin(), pos_y.cos()),
        )

        sin = torch.cat((sin_x, sin_y), dim=-1)
        cos = torch.cat((cos_x, cos_y), dim=-1)

        x = (x * cos) + (rotate_every_two(x) * sin)
        return x
