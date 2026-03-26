"""
Clase MultiheadAttention:
Este módulo implementa el mecanismo de atención multi-cabeza, una técnica clave en los transformadores. 
Permite que el modelo enfoque su atención en diferentes partes de la entrada simultáneamente, 
lo que mejora la capacidad de modelar relaciones complejas en los datos.

Atributos principales:
- `embed_dim`: Dimensión de los embeddings de entrada y salida.
- `num_heads`: Número de cabezas de atención. Más cabezas permiten capturar diferentes patrones de atención.
- `dropout`: Probabilidad de dropout aplicada a los pesos de atención.
- `bias_k` y `bias_v`: Parámetros opcionales para agregar sesgos a las claves y valores.
- `add_zero_attn`: Si es True, agrega vectores de atención cero para estabilizar el entrenamiento.

Este módulo se utiliza tanto para Self-Attention como para Cross-Attention en el archivo transformer_decoder.py.
"""

import warnings
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_

from .arm import AttentionRefinementModule


class MultiheadAttention(nn.Module):
    """Módulo de atención multi-cabeza usado en transformer_decoder.py.

    Este es el corazón del mecanismo de atención del transformador. Permite que el modelo
    enfoque su atención en diferentes partes de la entrada simultáneamente, capturando
    múltiples patrones de relaciones complejas entre tokens.

    La atención multi-cabeza funciona así: (1) Proyecta las entradas (Q, K, V) mediante
    capas lineales para crear representaciones independientes en cada cabeza; (2) Divide
    la dimensión de embedding en num_heads subcabezas; (3) Calcula scores de atención
    para cada cabeza usando Q·K^T / sqrt(d_k); (4) Aplica máscaras para evitar atender
    a padding o posiciones futuras inválidas; (5) Integra penalizaciones del ARM para
    Cross-Attention; (6) Combina las salidas de todas las cabezas mediante concatenación
    y una proyección lineal final. El uso de múltiples cabezas es más efectivo que una
    sola porque permite que el modelo aprender diferentes tipos de relaciones semánticas
    en paralelo (sintaxis, semántica, anáfora, etc.).
    """
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        kdim=None,
        vdim=None,
    ):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        # Detecta si Q, K, V provienen del mismo espacio de embedding (Self-Attention)
        # o de espacios diferentes (Cross-Attention)
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        # Cada cabeza de atención opera en un subespacio de dimensión head_dim
        # Por ejemplo, si embed_dim=512 y num_heads=8, head_dim=64
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        if self._qkv_same_embed_dim is False:
            # Cross-Attention: Q, K, V pueden tener dimensiones diferentes
            # Se almacenan como matrices de proyección separadas
            self.q_proj_weight = nn.Parameter(torch.Tensor(embed_dim, embed_dim))
            self.k_proj_weight = nn.Parameter(torch.Tensor(embed_dim, self.kdim))
            self.v_proj_weight = nn.Parameter(torch.Tensor(embed_dim, self.vdim))
            self.register_parameter("in_proj_weight", None)
        else:
            # Self-Attention: Q, K, V tienen la misma dimensión
            # Se proyectan usando una matriz única (3*embed_dim x embed_dim)
            # que será dividida en tres partes durante forward
            self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
            self.register_parameter("q_proj_weight", None)
            self.register_parameter("k_proj_weight", None)
            self.register_parameter("v_proj_weight", None)

        if bias:
            # Sesgos para las proyecciones de Q, K, V (3*embed_dim elementos)
            self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter("in_proj_bias", None)
        # Proyección lineal final que combina las salidas de todas las cabezas
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        if add_bias_kv:
            # Parámetros de sesgo opcionales para estabilizar el entrenamiento
            self.bias_k = nn.Parameter(torch.empty(1, 1, embed_dim))
            self.bias_v = nn.Parameter(torch.empty(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        # Agrega vectores de atención cero para mejorar la precisión de algunas tareas
        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.0)
            constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def __setstate__(self, state):
        # Support loading old MultiheadAttention checkpoints generated by v1.1.0
        if "_qkv_same_embed_dim" not in state:
            state["_qkv_same_embed_dim"] = True

        super(MultiheadAttention, self).__setstate__(state)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        arm: Optional[AttentionRefinementModule] = None,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        tgt_vocab:Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Calcula el output de atención multi-cabeza.

        Este método orquesta todo el cálculo de atención: proyecta Q, K, V en múltiples
        cabezas, calcula los pesos de atención, aplica máscaras y penalizaciones,
        y combina los resultados. Es usado desde transformer_decoder.py tanto para
        Self-Attention (query=key=value) como para Cross-Attention (query diferente de key/value).

        Parameters
        ----------
        query : Tensor
            Las consultas [tgt_len, batch_size, embed_dim]
        key : Tensor
            Las claves [src_len, batch_size, embed_dim o kdim]
        value : Tensor
            Los valores [src_len, batch_size, embed_dim o vdim]
        arm : AttentionRefinementModule, optional
            Módulo para refinar atención en Cross-Attention (aplica penalización del ARM)
        key_padding_mask : Tensor, optional
            Máscara [batch_size, src_len] para evitar atender a posiciones de padding
        need_weights : bool
            Si True, retorna los pesos de atención; si False, retorna None
        attn_mask : Tensor, optional
            Máscara [tgt_len, src_len] o [batch*num_heads, tgt_len, src_len]
            para evitar atender a posiciones futuras (causalidad)
        tgt_vocab : Tensor, optional
            Vocabulario objetivo para aplicar penalizaciones en Cross-Attention con ARM

        Returns
        -------
        attn_output : Tensor
            Output de atención [tgt_len, batch_size, embed_dim]
        attention : Tensor or None
            Pesos de atención [batch*num_heads, tgt_len, src_len] si need_weights=True
        """
        if not self._qkv_same_embed_dim:
            return multi_head_attention_forward(
                query,
                key,
                value,
                arm,
                self.embed_dim,
                self.num_heads,
                self.in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight,
                k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight,
                tgt_vocab=tgt_vocab,
            )
        else:
            return multi_head_attention_forward(
                query,
                key,
                value,
                arm,
                self.embed_dim,
                self.num_heads,
                self.in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                tgt_vocab=tgt_vocab,
            )


def multi_head_attention_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    arm: Optional[AttentionRefinementModule],
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight: Tensor,
    in_proj_bias: Tensor,
    bias_k: Optional[Tensor],
    bias_v: Optional[Tensor],
    add_zero_attn: bool,
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Tensor,
    training: bool = True,
    key_padding_mask: Optional[Tensor] = None,
    need_weights: bool = True,
    attn_mask: Optional[Tensor] = None,
    use_separate_proj_weight: bool = False,
    q_proj_weight: Optional[Tensor] = None,
    k_proj_weight: Optional[Tensor] = None,
    v_proj_weight: Optional[Tensor] = None,
    static_k: Optional[Tensor] = None,
    static_v: Optional[Tensor] = None,
    tgt_vocab:Optional[Tensor] = None,
) -> Tuple[Tensor, Optional[Tensor]]:
    tgt_len, bsz, embed_dim = query.size()
    assert embed_dim == embed_dim_to_check
    # allow MHA to have different sizes for the feature dimension
    assert key.size(0) == value.size(0) and key.size(1) == value.size(1)

    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
    # Escala de los productos internos: scaling = 1 / sqrt(head_dim)
    # Por ejemplo, si head_dim=64, scaling ≈ 0.125
    # Esta escala es crítica: previene que los scores de atención tengan valores
    # demasiado grandes (que causarían gradientes muy pequeños en backprop),
    # y mejora la estabilidad numérica del Softmax.
    # Sin este escalado, con head_dim grande, Q·K^T tendría valores enormes que
    # harían que Softmax colapsara a prácticamente ceros excepto el máximo.
    scaling = float(head_dim) ** -0.5

    # Proyección de entradas mediante capas lineales para obtener Q, K y V.
    # Este paso es crucial: transforma las representaciones de entrada en espacios
    # de consulta, clave y valor que permiten al modelo aprender qué atender.
    # Para Self-Attention: query=key=value, todas se proyectan igual
    # Para Cross-Attention: query diferente de key/value, proyecciones separadas
    if not use_separate_proj_weight:
        if (query is key or torch.equal(query, key)) and (
            key is value or torch.equal(key, value)
        ):
            # Self-Attention: query, key y value son idénticos
            # Una sola proyección lineal produce Q, K, V todos a la vez
            q, k, v = F.linear(query, in_proj_weight, in_proj_bias).chunk(3, dim=-1)

        elif key is value or torch.equal(key, value):
            # Cross-Attention: encoder-decoder attention
            # query proviene del decodificador, key/value del codificador
            # Se proyectan por separado usando diferentes ranuras de in_proj_weight
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = F.linear(query, _w, _b)

            if key is None:
                assert value is None
                k = None
                v = None
            else:

                # This is inline in_proj function with in_proj_weight and in_proj_bias
                _b = in_proj_bias
                _start = embed_dim
                _end = None
                _w = in_proj_weight[_start:, :]
                if _b is not None:
                    _b = _b[_start:]
                k, v = F.linear(key, _w, _b).chunk(2, dim=-1)

        else:
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = F.linear(query, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim
            _end = embed_dim * 2
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            k = F.linear(key, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim * 2
            _end = None
            _w = in_proj_weight[_start:, :]
            if _b is not None:
                _b = _b[_start:]
            v = F.linear(value, _w, _b)
    else:
        q_proj_weight_non_opt = torch.jit._unwrap_optional(q_proj_weight)
        len1, len2 = q_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == query.size(-1)

        k_proj_weight_non_opt = torch.jit._unwrap_optional(k_proj_weight)
        len1, len2 = k_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == key.size(-1)

        v_proj_weight_non_opt = torch.jit._unwrap_optional(v_proj_weight)
        len1, len2 = v_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == value.size(-1)

        if in_proj_bias is not None:
            q = F.linear(query, q_proj_weight_non_opt, in_proj_bias[0:embed_dim])
            k = F.linear(
                key, k_proj_weight_non_opt, in_proj_bias[embed_dim : (embed_dim * 2)]
            )
            v = F.linear(value, v_proj_weight_non_opt, in_proj_bias[(embed_dim * 2) :])
        else:
            q = F.linear(query, q_proj_weight_non_opt, in_proj_bias)
            k = F.linear(key, k_proj_weight_non_opt, in_proj_bias)
            v = F.linear(value, v_proj_weight_non_opt, in_proj_bias)
    # Aplica el escalado a Q inmediatamente después de proyectar
    # Esto prepara Q para la multiplicación con K^T en el siguiente paso
    q = q * scaling

    # Validación de tipos de attn_mask: solo float32/64/16, uint8 o bool
    # uint8 es deprecated, se convierte a bool automáticamente
    if attn_mask is not None:
        assert (
            attn_mask.dtype == torch.float32
            or attn_mask.dtype == torch.float64
            or attn_mask.dtype == torch.float16
            or attn_mask.dtype == torch.uint8
            or attn_mask.dtype == torch.bool
        ), "Only float, byte, and bool types are supported for attn_mask, not {}".format(
            attn_mask.dtype
        )
        if attn_mask.dtype == torch.uint8:
            warnings.warn(
                "Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead."
            )
            attn_mask = attn_mask.to(torch.bool)

        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0)
            if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                raise RuntimeError("The size of the 2D attn_mask is not correct.")
        elif attn_mask.dim() == 3:
            if list(attn_mask.size()) != [bsz * num_heads, query.size(0), key.size(0)]:
                raise RuntimeError("The size of the 3D attn_mask is not correct.")
        else:
            raise RuntimeError(
                "attn_mask's dimension {} is not supported".format(attn_mask.dim())
            )
        # attn_mask's dim is 3 now.

    # convert ByteTensor key_padding_mask to bool
    if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
        warnings.warn(
            "Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead."
        )
        key_padding_mask = key_padding_mask.to(torch.bool)

    if bias_k is not None and bias_v is not None:
        if static_k is None and static_v is None:
            k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = F.pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = F.pad(key_padding_mask, (0, 1))
        else:
            assert static_k is None, "bias cannot be added to static key."
            assert static_v is None, "bias cannot be added to static value."
    else:
        assert bias_k is None
        assert bias_v is None

    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    # Reorganiza las dimensiones para procesamiento multi-cabeza:
    # Transforma Q/K/V de [seq_len, batch, embed_dim] en [batch*num_heads, seq_len, head_dim]
    # Esto permite que cada cabeza procese su subespacio de dimensión head_dim de forma paralela.
    # Por ejemplo con batch=2, num_heads=8, head_dim=64, seq_len=10:
    #   Antes: [10, 2, 512]
    #   Después: [16, 10, 64] (16 = 2*8 cabezas)
    # Las capas posteriores de torch.bmm operan en paralelo sobre estas 16 matrices 10x64
    if k is not None:
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    if v is not None:
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

    if static_k is not None:
        assert static_k.size(0) == bsz * num_heads
        assert static_k.size(2) == head_dim
        k = static_k

    if static_v is not None:
        assert static_v.size(0) == bsz * num_heads
        assert static_v.size(2) == head_dim
        v = static_v

    src_len = k.size(1)

    if key_padding_mask is not None:
        assert key_padding_mask.size(0) == bsz
        assert key_padding_mask.size(1) == src_len

    if add_zero_attn:
        # Agrega vectores de atención cero para mejorar estabilidad
        src_len += 1
        k = torch.cat(
            [
                k,
                torch.zeros(
                    (k.size(0), 1) + k.size()[2:], dtype=k.dtype, device=k.device
                ),
            ],
            dim=1,
        )
        v = torch.cat(
            [
                v,
                torch.zeros(
                    (v.size(0), 1) + v.size()[2:], dtype=v.dtype, device=v.device
                ),
            ],
            dim=1,
        )
        if attn_mask is not None:
            attn_mask = F.pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = F.pad(key_padding_mask, (0, 1))

    # Cálculo de scores de atención: scores = Q · K^T / sqrt(d_k)
    # Esto genera una matriz [batch*num_heads, tgt_len, src_len] donde cada entrada
    # (i, j, k) representa la similitud entre el token j en la query y token k en la key
    # Para una cabeza con Q[16, 10, 64] y K[16, 15, 64], obtenemos [16, 10, 15]
    attn_output_weights = torch.bmm(q, k.transpose(1, 2))
    assert list(attn_output_weights.size()) == [bsz * num_heads, tgt_len, src_len]

    def mask_softmax_dropout(dots):
        # Aplica máscaras ANTES del Softmax para evitar que el modelo atienda a posiciones inválidas
        if attn_mask is not None:
            # attn_mask previene la atención al padding y a posiciones futuras (causalidad)
            # Usa -inf como valor de máscara para máscaras booleanas (bool)
            # O suma el valor de la máscara directamente para máscaras aditivas (float)
            if attn_mask.dtype == torch.bool:
                # print(dots)
                # Máscara booleana: donde attn_mask es True, pone -inf en scores
                dots.masked_fill_(attn_mask, float("-inf"))
                # print(dots)
            else:
                # Máscara aditiva (valores numéricos): suma directamente a los scores
                # Esto permite penalizaciones suaves en lugar de prohibiciones duras
                dots += attn_mask
        # print(dots)
        if key_padding_mask is not None:
            # key_padding_mask evita atender a posiciones de padding del codificador
            # Se reordena para aplicarse a cada cabeza por separado
            dots = dots.view(bsz, num_heads, tgt_len, src_len)
            dots = dots.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float("-inf"),
            )
            dots = dots.view(bsz * num_heads, tgt_len, src_len)
        # print(dots)
        # Softmax convierte los scores en probabilidades (suma a 1 en la dimensión src_len)
        # Después de aplicar -inf en máscaras, esas posiciones tendrán prob 0
        attn = F.softmax(dots, dim=-1)
        # print(attn)
        # Dropout para regularización durante entrenamiento
        attn = F.dropout(attn, p=dropout_p, training=training)
        return attn

    attention = mask_softmax_dropout(attn_output_weights)
    # print(attention)
    if arm is not None:
        # ARM (Attention Refinement Module) aplica una penalización en Cross-Attention
        # Resta el penalty del ARM a los scores ORIGINALES (attn_output_weights)
        # Lo que hace es desalentar la atención a ciertos tokens según tgt_vocab
        # Luego recalcula Softmax + Dropout con los scores penalizados
        attn_output_weights -= arm(attention, tgt_vocab)
        # Recalcula la atención con los scores penalizados
        attention = mask_softmax_dropout(attn_output_weights)

    # Combina los pesos de atención con los valores:
    # Para cada cabeza, multiplica [batch*num_heads, tgt_len, src_len] × [batch*num_heads, src_len, head_dim]
    # Resultado: [batch*num_heads, tgt_len, head_dim]
    # Cada posición en la salida es una suma ponderada de los valores según los pesos de atención
    attn_output = torch.bmm(attention, v)
    assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]
    # Reordena las dimensiones de vuelta a [tgt_len, batch, embed_dim]
    # Concatena las salidas de todas las cabezas
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    # Proyección lineal final que combina las representaciones de todas las cabezas
    # en un único espacio de salida de dimensión embed_dim
    attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)

    if need_weights:
        return attn_output, attention
    else:
        return attn_output, None
