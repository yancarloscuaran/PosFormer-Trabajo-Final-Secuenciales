from abc import abstractmethod
from typing import List, Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from Pos_Former.datamodule import vocab, vocab_size
from Pos_Former.utils.utils import Hypothesis, ce_loss, to_tgt_output
from einops import rearrange
from einops.einops import repeat
from torch import FloatTensor, LongTensor

from .beam_search import BeamSearchScorer

# modificado desde
# https://github.com/huggingface/transformers/blob/af6e01c5bc39467f1e3ce47a2135fb1777af1db2/src/transformers/generation_utils.py#L1843


class DecodeModel(pl.LightningModule):
    """
    Clase abstracta que define la interfaz base para modelos de decodificación.
    Esta clase debe ser heredada por modelos específicos como Decoder y PosDecoder
    (definidos en decoder.py) que implementan el método abstracto transform().
    Durante la inferencia, lit_posformer.py instancia estas clases y utiliza el
    método beam_search() para generar secuencias de tokens autorregressivamente.
    """
    @abstractmethod
    def transform(
        self, src: List[FloatTensor], src_mask: List[LongTensor], input_ids: LongTensor
    ) -> FloatTensor:
        """
        Método abstracto que realiza un paso del decodificador transformador.
        Las subclases (Decoder y PosDecoder) implementan esta función para procesar
        los embeddings de entrada de la imagen (src) y calcular distribuciones de
        probabilidad sobre el vocabulario para la siguiente posición.

        Parameters
        ----------
        src : List[FloatTensor]
            características extraídas del codificador [b, t, d]
        src_mask : List[LongTensor]
            máscara para indicar posiciones válidas [b, t]
        input_ids : LongTensor
            IDs de los tokens ya generados [b, l]

        Returns
        -------
        FloatTensor
            logits para todos los tokens en el vocabulario [b, l, vocab_size]
        """
        raise NotImplementedError("This is an abstract method.")

    def beam_search(
        self,
        src: List[FloatTensor],
        src_mask: List[LongTensor],
        beam_size: int,
        max_len: int,
        alpha: float,
        early_stopping: bool,
        temperature: float,
    ) -> List[Hypothesis]:
        """
        Ejecuta Beam Search para generar secuencias de tokens mediante decodificación autoregresiva.
        El Beam Search es un algoritmo que, en lugar de seleccionar siempre el token más probable
        (greedy decoding), mantiene un conjunto de K hipótesis parciales candidatas y las expande
        simultáneamente. Esto permite explorar un espacio de búsqueda más amplio y encontrar
        secuencias globalmente óptimas en lugar de caer en óptimos locales pobres.

        En este modelo se implementa Beam Search bidireccional: genera dos secuencias en paralelo,
        una de izquierda a derecha (l2r) y otra de derecha a izquierda (r2l), luego combina
        ambas predicciones para obtener una mejor representación de la ecuación matemática.

        Parameters
        ----------
        src : List[FloatTensor]
            características visuales del codificador [b, t, d]
        src_mask : List[LongTensor]
            máscara binaria para posiciones válidas [b, t]
        beam_size : int
            número de hipótesis a mantener activas. Mayor beam_size mejora la calidad pero
            incrementa el costo computacional. Típicamente valores entre 5-10 balancean
            calidad y eficiencia.
        max_len : int
            longitud máxima de las secuencias generadas
        alpha : float
            coeficiente de penalización de longitud que previene favorecer secuencias cortas.
            Se aplica como ^alpha donde es el número de tokens no-padding. Un alpha mayor
            penaliza más las secuencias cortas, incrementando la longitud promedio.
        early_stopping : bool
            si es Verdadero, detiene la búsqueda cuando encuentra una secuencia completa
            (con token EOS) con puntuación superior a todos los prefijos restantes.
            Esto acelera la inferencia sin afectar significativamente la calidad.
        temperature : float
            hiperparámetro que controla la aleatoriedad de las distribuciones de probabilidad.
            Valores mayores (>1) hacen las distribuciones más uniformes; valores menores (<1)
            concentran probabilidad en los tokens más probables.

        Returns
        -------
        List[Hypothesis]: lista con batch_size hipótesis, cada una contiene la secuencia
                         generada, su puntuación normalizada y su dirección.
        """
        batch_size = src[0].shape[0] * 2  # mul 2 for bi-direction
        batch_beam_size = batch_size * beam_size
        half_bb_size = batch_beam_size // 2

        for i in range(len(src)):
            # [2 * b, t, d], [l2r l2r, r2l r2l]
            src[i] = torch.cat((src[i], src[i]), dim=0)
            src_mask[i] = torch.cat((src_mask[i], src_mask[i]), dim=0)

        l2r = torch.full(
            (batch_size // 2, 1),
            fill_value=vocab.SOS_IDX,
            dtype=torch.long,
            device=self.device,
        )
        r2l = torch.full(
            (batch_size // 2, 1),
            fill_value=vocab.EOS_IDX,
            dtype=torch.long,
            device=self.device,
        )
        input_ids = torch.cat((l2r, r2l), dim=0)

        beam_scorer = BeamSearchScorer(
            batch_size, beam_size, alpha, early_stopping, self.device
        )

        # Ejecuta la búsqueda bidireccional principal que expande hipótesis autorregressivamente
        hyps, scores = self._beam_search(
            src=src,
            src_mask=src_mask,
            input_ids=input_ids,
            beam_scorer=beam_scorer,
            beam_size=beam_size,
            max_len=max_len,
            temperature=temperature,
        )
        # Invierte las hipótesis generadas de derecha-a-izquierda para alinearlas en dirección
        for i in range(half_bb_size, batch_beam_size):
            hyps[i] = torch.flip(hyps[i], dims=[0])     # flip反转

        # Prepara las hipótesis para re-puntuar: agrega tokens de inicio y hace padding
        lens = [len(h) + 1 for h in hyps]  # plus to append start token
        r2l_tgt, r2l_out = to_tgt_output(
            hyps[:half_bb_size], "r2l", self.device, pad_to_len=max(lens)
        )
        l2r_tgt, l2r_out = to_tgt_output(
            hyps[half_bb_size:], "l2r", self.device, pad_to_len=max(lens)
        )
        tgt = torch.cat((l2r_tgt, r2l_tgt), dim=0)
        out = torch.cat((l2r_out, r2l_out), dim=0)

        # Re-calcula puntuaciones usando el modelo completo con normalización por longitud
        # Esta segunda pasada de puntuación favorece predicciones coherentes globalmente
        rev_scores = self._rate(src, src_mask, tgt, out, alpha, temperature)
        rev_scores = torch.cat(
            (rev_scores[half_bb_size:], rev_scores[:half_bb_size]), dim=0
        )
        # Combina puntuaciones de ambas direcciones para obtener puntuación final
        scores = scores + rev_scores

        # Reorganiza puntuaciones para identificar la mejor hipótesis entre ambas direcciones
        scores = rearrange(scores, "(b m) -> b m", b=batch_size)
        l2r_scores, r2l_scores = torch.chunk(scores, 2, dim=0)
        # Concatena para comparar todas las hipótesis de ambas direcciones simultáneamente
        scores = torch.cat((l2r_scores, r2l_scores), dim=1)
        # Selecciona la hipótesis con mayor puntuación final (combinada y normalizada por longitud)
        best_scores, best_indices = torch.max(scores, dim=1)
        best_split = best_indices // beam_size  # Determina si vino de l2r o r2l
        best_indices = best_indices % beam_size  # Índice dentro de las beam_size hipótesis
        batch_indices = torch.arange(
            0, batch_size // 2, dtype=torch.long, device=self.device
        )
        best_indices = (
            best_split * half_bb_size + batch_indices * beam_size + best_indices
        )

        # Construye lista de hipótesis finales con sus puntuaciones normalizadas
        ret: List[Hypothesis] = []
        for idx, score in zip(best_indices, best_scores):
            hpy = Hypothesis(hyps[idx], score, "l2r")
            ret.append(hpy)
        return ret

    def _beam_search(
        self,
        src: List[FloatTensor],
        src_mask: List[LongTensor],
        input_ids: LongTensor,
        beam_scorer: BeamSearchScorer,
        beam_size: int,
        max_len: int,
        temperature: float,
    ) -> Tuple[List[LongTensor], FloatTensor]:
        """
        Implementación interna del Beam Search que expande iterativamente las hipótesis.
        En cada paso: obtiene logits del modelo, calcula log-probabilidades, combina
        con puntuaciones previas, selecciona los top-2K candidatos, y mantiene los
        K mejores usando el gestor de puntuación (BeamSearchScorer).
        
        El proceso se detiene cuando se alcanza max_len o cuando early_stopping
        detecta que ninguna hipótesis incompleta puede superar la mejor hipótesis
        completa encontrada hasta ahora.

        Parameters
        ----------
        src : List[FloatTensor]
            características visuales [b, t, d]
        src_mask : List[LongTensor]
            máscara de posiciones válidas [b, t]
        input_ids: LongTensor
            tokens generados hasta ahora [b, 1] al inicio
        beam_scorer : BeamSearchScorer
            gestor que implementa la lógica de selección y filtrado de hipótesis
        beam_size : int
            número de hipótesis a mantener en cada paso
        max_len : int
            límite máximo de pasos de generación
        temperature : float
            parámetro para controlar aleatoriedad en distribuciones

        Returns
        _______
        Tuple[List[LongTensor], FloatTensor]
            List[LongTensor]: [b * beam_size] hipótesis sin tokens SOS/EOS
            FloatTensor: [b * beam_size] puntuaciones acumuladas normalizadas
        """

        batch_size, cur_len = input_ids.shape

        beam_scores = torch.zeros(batch_size, dtype=torch.float, device=self.device)
        # Itera hasta alcanzar longitud máxima o detectar convergencia mediante early_stopping
        while cur_len < max_len and not beam_scorer.is_done():
            # Obtiene logits del siguiente token usando solo el último paso de salida del transformador
            next_token_logits  = (
                self.transform(src, src_mask, input_ids)[:, -1, :] / temperature
            )
            # Convierte logits a log-probabilidades usando softmax
            next_token_scores = F.log_softmax(next_token_logits, dim=-1)

            # Suma puntuaciones previas (acumuladas) para obtener puntuación total de cada hipótesis
            # Esto favorece hipótesis que ya eran buenas en pasos anteriores
            next_token_scores = next_token_scores + beam_scores[:, None].expand_as(
                next_token_scores
            )
            # Reorganiza para considerar conjuntamente todas las expansiones posibles
            reshape_size = next_token_scores.shape[0] // batch_size
            next_token_scores = rearrange(
                next_token_scores,
                "(b m) v -> b (m v)",
                m=reshape_size,
            )

            # Selecciona los 2*beam_size mejores candidatos (permite eliminar los K peores después)
            next_token_scores, next_tokens = torch.topk(
                next_token_scores, 2 * beam_size, dim=1
            )

            # Descodifica índices: calcula qué hipótesis anterior y qué token nuevo
            next_indices = next_tokens // vocab_size  # Índice de hipótesis anterior (0 a beam_size-1)
            next_tokens = next_tokens % vocab_size     # ID del token nuevo en vocabulario

            # En el primer paso, expande el batch del token inicial a beam_size copias
            # para que las hipótesis tengan puntuaciones independientes desde el principio
            if cur_len == 1:
                input_ids = repeat(input_ids, "b l -> (b m) l", m=beam_size)
                for i in range(len(src)):
                    src[i] = repeat(src[i], "b ... -> (b m) ...", m=beam_size)
                    src_mask[i] = repeat(src_mask[i], "b ... -> (b m) ...", m=beam_size)

            # Aplica lógica de Beam Search: mantiene solo las beam_size mejores hipótesis
            # y descarta las demás para ahorrar memoria y computación en próximos pasos
            beam_scores, beam_next_tokens, beam_idx = beam_scorer.process(
                input_ids=input_ids,
                next_scores=next_token_scores,
                next_tokens=next_tokens,
                next_indices=next_indices,
            )
            # Construye nuevas secuencias: toma las hipótesis seleccionadas y agrega el nuevo token
            input_ids = torch.cat(
                (input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)), dim=-1
            )

            cur_len += 1
        # Finalizador extrae hipótesis completas, filtra padding y retorna puntuaciones
        return beam_scorer.finalize(input_ids, beam_scores)

    def _rate(
        self,
        src: List[FloatTensor],
        src_mask: List[LongTensor],
        tgt: LongTensor,
        out: LongTensor,
        alpha: float,
        temperature: float,
    ) -> FloatTensor:
        """
        Re-puntúa hipótesis completas usando el modelo completo en lugar de puntuaciones
        acumuladas paso a paso. Esta segunda pasada normaliza las puntuaciones por longitud
        usando el parámetro alpha, favoreciendo explicaciones más balanceadas.

        Parameters
        ----------
        src : List[FloatTensor]
            características visuales expandidas [b * beam_size, t, d]
        src_mask : List[LongTensor]
            máscara de validez expandida [b * beam_size, t]
        tgt : LongTensor
            hipótesis completas con token inicial [b * beam_size, l]
        out : LongTensor
            versión desplazada para calcular pérdida [b * beam_size, l]
        alpha : float
            exponente para penalización de longitud. Mayor alpha penaliza secuencias cortas
        temperature : float
            parámetro de aleatoriedad aplicado también en re-puntuación

        Returns
        -------
        FloatTensor
            puntuaciones normalizadas por longitud [b * beam_size]
        """
        b = tgt.shape[0]

        # Procesa secuencias completas a través del modelo transformador
        out_hat = self.transform(src, src_mask, tgt) / temperature

        # Calcula pérdida (log-probabilidad negativa) token por token sin reducción
        loss = ce_loss(out_hat, out, reduction="none")
        loss = rearrange(loss, "(b l) -> b l", b=b)

        # Aplica penalización de longitud: secuencias sin padding se cuenta, el resto se ignora
        mask = tgt == vocab.PAD_IDX
        # Penaliza inversamente proporcional a (longitud)^alpha para favorecer longitudes naturales
        penalty = (~mask).sum(dim=1) ** alpha
        # Promedia pérdida y normaliza: menor valor = mejor hipótesis
        loss = -torch.sum(loss, dim=1) / penalty

        return loss
class PosDecodeModel(pl.LightningModule):
    """
    Variante del modelo de decodificación que implementa funcionalidad similar a DecodeModel
    pero con capacidad adicional para retornar pesos de atención (attn). Hereda esta clase
    Decoder posicional o extensiones que necesiten exponer la estructura de atención del modelo.
    Se utiliza de la misma forma durante la inferencia desde lit_posformer.py.
    """
    @abstractmethod
    def transform(
        self, src: List[FloatTensor], src_mask: List[LongTensor], input_ids: LongTensor
    ) -> FloatTensor:
        """
        Método abstracto para decodificación autorregresiva (ver DecodeModel.transform).
        Las subclases heredan esta interfaz y la implementan según su arquitectura específica.

        Parameters
        ----------
        src : List[FloatTensor]
            características visuales [b, t, d]
        src_mask : List[LongTensor]
            máscara binaria [b, t]
        input_ids : LongTensor
            secuencia de entrada [b, l]

        Returns
        -------
        FloatTensor
            logits [b, l, vocab_size]
        """
        raise NotImplementedError("This is an abstract method.")

    def beam_search(
        self,
        src: List[FloatTensor],
        src_mask: List[LongTensor],
        beam_size: int,
        max_len: int,
        alpha: float,
        early_stopping: bool,
        temperature: float,
    ) -> List[Hypothesis]:
        """
        Ejecuta Beam Search bidireccional similar a DecodeModel, pero con soporte adicional
        para capturar pesos de atención durante la generación. Utilizado durante la inferencia
        desde lit_posformer.py cuando se desea analizar qué partes de la imagen atiende el modelo.
        Ver DecodeModel.beam_search para detalles del algoritmo.

        Parameters
        ----------
        src : List[FloatTensor]
            características [b, t, d]
        src_mask : List[LongTensor]
            máscara [b, t]
        beam_size : int
            número de hipótesis simultáneas
        max_len : int
            máxima longitud de generación
        alpha : float
            penalización de longitud
        early_stopping : bool
            activar parada temprana de búsqueda
        temperature : float
            control de aleatoriedad

        Returns
        -------
        List[Hypothesis]: lista de hipótesis [batch_size,]
        """
        # Inicia búsqueda bidireccional: duplica batch para l2r y r2l
        batch_size = src[0].shape[0] * 2  # mul 2 for bi-direction
        batch_beam_size = batch_size * beam_size
        half_bb_size = batch_beam_size // 2

        # Duplica características y máscaras para procesamiento simultáneo
        for i in range(len(src)):
            # [2 * b, t, d], [l2r l2r, r2l r2l]
            src[i] = torch.cat((src[i], src[i]), dim=0)
            src_mask[i] = torch.cat((src_mask[i], src_mask[i]), dim=0)

        # Inicializa tokens de inicio según dirección
        l2r = torch.full(
            (batch_size // 2, 1),
            fill_value=vocab.SOS_IDX,
            dtype=torch.long,
            device=self.device,
        )
        r2l = torch.full(
            (batch_size // 2, 1),
            fill_value=vocab.EOS_IDX,
            dtype=torch.long,
            device=self.device,
        )
        input_ids = torch.cat((l2r, r2l), dim=0)

        # Gestor de Beam Search que mantiene las K mejores hipótesis por paso
        beam_scorer = BeamSearchScorer(
            batch_size, beam_size, alpha, early_stopping, self.device
        )

        # Ejecuta búsqueda principal con posibilidad de capturar atención
        hyps, scores = self._beam_search(
            src=src,
            src_mask=src_mask,
            input_ids=input_ids,
            beam_scorer=beam_scorer,
            beam_size=beam_size,
            max_len=max_len,
            temperature=temperature,
        )
        # Invierte hipótesis r2l a dirección canónica
        for i in range(half_bb_size, batch_beam_size):
            hyps[i] = torch.flip(hyps[i], dims=[0])     #flip反转

        # Prepara secuencias para re-puntuación: agrega tokens iniciales y padding
        lens = [len(h) + 1 for h in hyps]  # plus to append start token
        r2l_tgt, r2l_out = to_tgt_output(
            hyps[:half_bb_size], "r2l", self.device, pad_to_len=max(lens)
        )
        l2r_tgt, l2r_out = to_tgt_output(
            hyps[half_bb_size:], "l2r", self.device, pad_to_len=max(lens)
        )
        tgt = torch.cat((l2r_tgt, r2l_tgt), dim=0)
        out = torch.cat((l2r_out, r2l_out), dim=0)

        # Re-calcula puntuaciones con normalización por longitud
        rev_scores = self._rate(src, src_mask, tgt, out, alpha, temperature)
        rev_scores = torch.cat(
            (rev_scores[half_bb_size:], rev_scores[:half_bb_size]), dim=0
        )
        # Suma puntuaciones de ambas direcciones
        scores = scores + rev_scores

        # Reorganiza para comparar hipótesis de ambas direcciones
        scores = rearrange(scores, "(b m) -> b m", b=batch_size)
        l2r_scores, r2l_scores = torch.chunk(scores, 2, dim=0)
        scores = torch.cat((l2r_scores, r2l_scores), dim=1)
        # Selecciona mejor hipótesis (combinada y normalizada)
        best_scores, best_indices = torch.max(scores, dim=1)
        best_split = best_indices // beam_size  # De qué dirección vino
        best_indices = best_indices % beam_size  # Índice en esa dirección
        batch_indices = torch.arange(
            0, batch_size // 2, dtype=torch.long, device=self.device
        )
        best_indices = (
            best_split * half_bb_size + batch_indices * beam_size + best_indices
        )

        # Construye lista final con hipótesis seleccionadas
        ret: List[Hypothesis] = []
        for idx, score in zip(best_indices, best_scores):
            hpy = Hypothesis(hyps[idx], score, "l2r")
            ret.append(hpy)
        return ret, attn

    def _beam_search(
        self,
        src: List[FloatTensor],
        src_mask: List[LongTensor],
        input_ids: LongTensor,
        beam_scorer: BeamSearchScorer,
        beam_size: int,
        max_len: int,
        temperature: float,
    ) -> Tuple[List[LongTensor], FloatTensor]:
        """
        Implementación interna del Beam Search (similar a DecodeModel._beam_search).
        Mantiene K hipótesis activas paralelamente, expande cada una con los tokens
        más probables, y descarta las peores en cada paso para mantener eficiencia.
        Ver DecodeModel._beam_search para detalles algorítmicos completos.

        Parameters
        ----------
        src : List[FloatTensor]
            características visuales [b, t, d]
        src_mask : List[LongTensor]
            máscara de secuencia [b, t]
        input_ids: LongTensor
            tokens iniciales [b, 1]
        beam_scorer : BeamSearchScorer
            gestor de selección de hipótesis
        beam_size : int
            número de hipótesis a mantener
        max_len : int
            límite máximo de pasos
        temperature : float
            parámetro de aleatoriedad

        Returns
        _______
        Tuple[List[LongTensor], FloatTensor]
            List[LongTensor]: hipótesis [b * beam_size]
            FloatTensor: puntuaciones normalizadas [b * beam_size]
        """

        batch_size, cur_len = input_ids.shape

        # Acumula log-probabilidades de hipótesis
        beam_scores = torch.zeros(batch_size, dtype=torch.float, device=self.device)
        # Expande iterativamente hasta convergencia o max_len
        while cur_len < max_len and not beam_scorer.is_done():
            # Procesa último token para obtener distribución siguiente
            next_token_logits  = (
                self.transform(src, src_mask, input_ids)[:, -1, :] / temperature
            )
            # Normaliza a probabilidades logarítmicas
            next_token_scores = F.log_softmax(next_token_logits, dim=-1)

            # Combina con puntuaciones previas para score total de cada camino
            next_token_scores = next_token_scores + beam_scores[:, None].expand_as(
                next_token_scores
            )
            # Reorganiza para evaluar todas las expansiones posibles conjuntamente
            reshape_size = next_token_scores.shape[0] // batch_size
            next_token_scores = rearrange(
                next_token_scores,
                "(b m) v -> b (m v)",
                m=reshape_size,
            )

            # Selecciona 2*K mejores para poder descartar K peores después
            next_token_scores, next_tokens = torch.topk(
                next_token_scores, 2 * beam_size, dim=1
            )

            # Descodifica qué hipótesis anterior y qué token nuevo
            next_indices = next_tokens // 6  # Hipótesis anterior (para vocabulario=6)
            next_tokens = next_tokens % 6      # Token en vocabulario

            if cur_len == 1:
                input_ids = repeat(input_ids, "b l -> (b m) l", m=beam_size)
                for i in range(len(src)):
                    src[i] = repeat(src[i], "b ... -> (b m) ...", m=beam_size)
                    src_mask[i] = repeat(src_mask[i], "b ... -> (b m) ...", m=beam_size)

            beam_scores, beam_next_tokens, beam_idx = beam_scorer.process(
                input_ids=input_ids,
                next_scores=next_token_scores,
                next_tokens=next_tokens,
                next_indices=next_indices,
            )
            #print("input_ids")
            #print(input_ids)  # 测试
            input_ids = torch.cat(
                (input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)), dim=-1
            )

            cur_len += 1
        return beam_scorer.finalize(input_ids, beam_scores)

    def _rate(
        self,
        src: List[FloatTensor],
        src_mask: List[LongTensor],
        tgt: LongTensor,
        out: LongTensor,
        alpha: float,
        temperature: float,
    ) -> FloatTensor:
        """
        Re-puntúa hipótesis completas con normalización por longitud.
        Ver DecodeModel._rate para descripción detallada del algoritmo.

        Parameters
        ----------
        src : List[FloatTensor]
            características visuales [b * beam_size, t, d]
        src_mask : List[LongTensor]
            máscara de validez [b * beam_size, t]
        tgt : LongTensor
            hipótesis con tokens iniciales [b * beam_size, l]
        out : LongTensor
            versión desplazada para pérdida [b * beam_size, l]
        alpha : float
            exponente para penalización de longitud
        temperature : float
            parámetro de aleatoriedad en distribuciones

        Returns
        -------
        FloatTensor
            puntuaciones normalizadas [b * beam_size]
        """
        b = tgt.shape[0]

        # Evalúa secuencias completas en modelo
        out_hat = self.transform(src, src_mask, tgt) / temperature

        # Calcula pérdida token a token
        loss = ce_loss(out_hat, out, reduction="none")
        loss = rearrange(loss, "(b l) -> b l", b=b)

        # Normaliza por longitud: penaliza secuencias según (longitud)^alpha
        mask = tgt == vocab.PAD_IDX
        penalty = (~mask).sum(dim=1) ** alpha
        loss = -torch.sum(loss, dim=1) / penalty

        return loss