from typing import List, Tuple

import torch
from Pos_Former.datamodule import vocab
from torch import FloatTensor, LongTensor


# modificado desde
# https://github.com/huggingface/transformers/blob/af6e01c5bc39467f1e3ce47a2135fb1777af1db2/src/transformers/generation_beam_search.py#L206
class BeamSearchScorer:
    """
    Gestor de estado para Beam Search que mantiene un registro de hipótesis candidatas.
    Coordina la selección de las K mejores hipótesis en cada paso de decodificación
    autorregressiva. Utilizado desde generation_utils.py durante la inferencia.
    
    Este scorer mantiene una lista de BeamHypotheses (una por elemento del batch)
    y rastrea cuál hipótesis está completa y lista para su puntuación final.
    """
    def __init__(
        self,
        batch_size: int,
        beam_size: int,
        alpha: float,
        do_early_stopping: bool,
        device: torch.device,
    ) -> None:
        self.batch_size = batch_size
        self.beam_size = beam_size  # Número de hipótesis a mantener activas
        self.alpha = alpha  # Exponente para penalización de longitud
        self.device = device

        # Crea un gestor de hipótesis para cada elemento del batch
        # Cada uno mantiene las beam_size mejores hipótesis completadas
        self._beam_hyps = [
            BeamHypotheses(beam_size, alpha, do_early_stopping)
            for _ in range(batch_size)
        ]

        # Rastrea si la búsqueda está completa para cada elemento del batch
        # Se marca como True cuando el scorer detecta que no pueden aparecer mejores hipótesis
        self._done = torch.tensor(
            [False for _ in range(batch_size)], dtype=torch.bool, device=self.device
        )

    def is_done(self) -> bool:
        """Verifica si la búsqueda está completa para todos los elementos del batch."""
        return self._done.all()

    def process(
        self,
        input_ids: LongTensor,
        next_scores: FloatTensor,
        next_tokens: LongTensor,
        next_indices: LongTensor,
    ) -> Tuple[FloatTensor, LongTensor, LongTensor]:
        """
        Procesa los mejores candidatos del paso actual y selecciona las beam_size hipótesis
        más prometedoras para continuar. Detecta hipótesis completas (con token EOS o SOS
        dependiendo de la dirección de decodificación) y las pasa a su gestor de hipótesis.
        Las hipótesis incompletas se retornan para expandirse en el siguiente paso.
        
        El método mantiene actualizado el score acumulado de cada hipótesis parcial,
        que será usado en el siguiente paso de decodificación.

        Parameters
        ----------
        input_ids : LongTensor
            Secuencias parciales generadas hasta ahora [b * beam_size, l]
        next_scores : FloatTensor
            Log-probabilidades de los 2*beam_size mejores candidatos [b, 2 * beam_size]
        next_tokens : LongTensor
            IDs de esos tokens candidatos [b, 2 * beam_size]
        next_indices : LongTensor
            Índices de cuál hipótesis anterior fue expandida [b, 2 * beam_size]

        Returns
        -------
        Tuple[FloatTensor, LongTensor, LongTensor]
            next_scores: Scores de las beam_size hipótesis seleccionadas [b * beam_size]
            next_tokens: Tokens de esas hipótesis [b * beam_size]
            next_indices: Índices de padres para mantener trazabilidad [b * beam_size]
        """
        # Almacena las beam_size mejores hipótesis no-completadas de este paso
        next_beam_scores = torch.zeros(
            (self.batch_size, self.beam_size),
            dtype=next_scores.dtype,
            device=self.device,
        )
        next_beam_tokens = torch.zeros(
            (self.batch_size, self.beam_size),
            dtype=next_tokens.dtype,
            device=self.device,
        )
        next_beam_indices = torch.zeros(
            (self.batch_size, self.beam_size),
            dtype=next_indices.dtype,
            device=self.device,
        )

        # Itera sobre cada elemento del batch procesando sus hipótesis candidatas
        for batch_idx, beam_hyp in enumerate(self._beam_hyps):
            if self._done[batch_idx]:
                # Si la búsqueda ya terminó para este batch, rellena con padding
                assert len(beam_hyp) >= self.beam_size
                next_beam_scores[batch_idx, :] = 0
                next_beam_tokens[batch_idx, :] = vocab.PAD_IDX
                next_beam_indices[batch_idx, :] = batch_idx * self.beam_size
                continue

            # Selecciona las beam_size mejores hipótesis candidatas para el siguiente paso
            beam_idx = 0
            for beam_token_rank, (next_score, next_token, next_index) in enumerate(
                zip(
                    next_scores[batch_idx],
                    next_tokens[batch_idx],
                    next_indices[batch_idx],
                )
            ):
                batch_beam_idx = batch_idx * self.beam_size + next_index
                # Verifica si la hipótesis está completa detectando tokens de terminación
                # Bidireccional: l2r termina con EOS después de SOS, r2l termina con SOS después de EOS
                l2r_done = (
                    input_ids[batch_beam_idx][0].item() == vocab.SOS_IDX
                    and next_token.item() == vocab.EOS_IDX
                )
                r2l_done = (
                    input_ids[batch_beam_idx][0].item() == vocab.EOS_IDX
                    and next_token.item() == vocab.SOS_IDX
                )
                if l2r_done or r2l_done:
                    # Hipótesis completada: envía al gestor para almacenarla entre las mejores
                    if beam_token_rank >= self.beam_size:
                        # Solo acepta completadas de los top-beam_size para evitar sesgos
                        continue
                    beam_hyp.add(input_ids[batch_beam_idx].clone(), next_score.item())
                else:
                    # Hipótesis incompleta: prepara para siguiente paso de expansión
                    next_beam_scores[batch_idx, beam_idx] = next_score
                    next_beam_tokens[batch_idx, beam_idx] = next_token
                    next_beam_indices[batch_idx, beam_idx] = batch_beam_idx
                    beam_idx += 1

                # Cuando se llenan las beam_size hipótesis, no procesa más candidatos
                if beam_idx == self.beam_size:
                    break

            assert beam_idx == self.beam_size

            # Verifica si la búsqueda puede terminar anticipadamente para este batch
            # (early_stopping) comparando el mejor score potencial con el peor completado
            self._done[batch_idx] = beam_hyp.is_done(
                best_sum_logprobs=next_beam_scores[batch_idx].max().item(),
                cur_len=input_ids.shape[-1],
            )

        # Retorna los scores, tokens e índices de las mejores beam_size hipótesis seleccionadas
        # (redondeado a dimensión batch)
        return (
            next_beam_scores.view(-1),
            next_beam_tokens.view(-1),
            next_beam_indices.view(-1),
        )

    def finalize(
        self,
        input_ids: LongTensor,
        final_scores: FloatTensor,
    ) -> Tuple[List[LongTensor], FloatTensor]:
        """Extrae las mejores beam_size hipótesis después de terminar la búsqueda.

        Convierte hipótesis incompletas (si se alcanzó max_len) en completadas,
        recupera los top num_beams según scored_logprobs normalizados,
        y retorna secuencias sin tokens SOS/EOS.

        Parameters
        ----------
        input_ids : LongTensor
            [b * beam_size, l] - todas las secuencias en el estado final
        final_scores : FloatTensor
            [b * beam_size] - scores de log-probabilidades acumuladas finales

        Returns
        -------
        Tuple[List[LongTensor], FloatTensor]
            List[LongTensor]: [b * beam_size] secuencias sin SOS ni EOS
            FloatTensor: [b * beam_size] scores normalizados correspondientes
        """
        # Convierte hipótesis incompletas (que alcanzaron max_len) en completadas
        # para que se almacenen en el gestor BeamHypotheses
        for batch_idx, beam_hyp in enumerate(self._beam_hyps):
            if self._done[batch_idx]:
                # Si ya se completó la búsqueda, sus mejores hipótesis ya están almacenadas
                continue

            # Procesa las beam_size hipótesis incompletas del batch actual
            # y las agrega al gestor que automáticamente mantiene los mejores num_beams
            for beam_id in range(self.beam_size):
                batch_beam_idx = batch_idx * self.beam_size + beam_id
                final_score = final_scores[batch_beam_idx].item()
                final_tokens = input_ids[batch_beam_idx]
                beam_hyp.add(final_tokens, final_score)

        # Extrae las mejores hipótesis de cada batch y sus scores
        all_hyps: List[LongTensor] = []
        scores: FloatTensor = torch.zeros(
            self.batch_size * self.beam_size, dtype=torch.float, device=self.device
        )

        # Recorre los gestionadores de hipótesis (uno por elemento del batch)
        for beam_hyp in self._beam_hyps:
            # Cada beam_hyp.beams está ordenado por score (mejor primero)
            # y contiene hasta num_beams mejores hipótesis completadas
            for score, seq in beam_hyp.beams:
                scores[len(all_hyps)] = score
                # Elimina el token SOS (primer token) para retornar solo los tokens generados
                all_hyps.append(seq[1:])

        return all_hyps, scores


class BeamHypotheses:
    """Gestor de hipótesis completadas para un elemento del batch.

    Mantiene un heap de las mejores num_beams hipótesis con scores normalizados
    por longitud. Cuando una hipótesis alcanza EOS/SOS (según dirección de decodificación),
    se añade a este gestor. Automáticamente retiene solo los mejores scores,
    descartando los de menor puntuación cuando se excede num_beams.
    """
    def __init__(self, num_beams: int, length_penalty: float, early_stopping: bool):
        """Inicializa gestor de las n-mejores hipótesis completadas.

        Parameters
        ----------
        num_beams : int
            Número máximo de hipótesis completadas a mantener
        length_penalty : float
            Exponente alpha para normalización de scores: score_norm = sum_logprobs / length^alpha
        early_stopping : bool
            Si True, permite terminar búsqueda antes de max_len comparando best vs worst
        """
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.num_beams = num_beams
        self.beams: List[Tuple[float, LongTensor]] = []
        self.worst_score = 1e9

    def __len__(self):
        """Retorna la cantidad de hipótesis completadas almacenadas."""
        return len(self.beams)

    def add(self, hyp: LongTensor, sum_logprobs: float):
        """Agrega una hipótesis completada al gestor, manteniendo los mejores num_beams.

        Normaliza el score usando length_penalty para evitar sesgo hacia secuencias cortas.
        Cuando se excede num_beams, descarta automáticamente el peor (menor score).

        Parameters
        ----------
        hyp : LongTensor
            Secuencia completada (tokenizada)
        sum_logprobs : float
            Suma de log-probabilidades acumuladas durante decodificación
        """
        # Normaliza score por longitud: penaliza o recompensa secuencias largas según alpha
        score = sum_logprobs / (hyp.shape[-1] ** self.length_penalty)
        
        # Acepta si: (1) aún no hay num_beams hipótesis, o (2) score es mejor que el peor almacenado
        if len(self) < self.num_beams or score > self.worst_score:
            self.beams.append((score, hyp))
            
            # Si excede num_beams, descarta el de menor score y actualiza worst_score
            if len(self) > self.num_beams:
                # Ordena por score (menor primero) para identificar el peor
                sorted_next_scores = sorted(
                    [(s, idx) for idx, (s, _) in enumerate(self.beams)]
                )
                # Elimina el peor (índice 0 en lista ordenada)
                del self.beams[sorted_next_scores[0][1]]
                # Actualiza worst_score al nuevo peor (índice 1 después de eliminar)
                self.worst_score = sorted_next_scores[1][0]
            else:
                # Si aún no llena, actualiza worst_score con el mínimo visto hasta ahora
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs: float, cur_len: int) -> bool:
        """Detecta si es posible terminar la búsqueda para este batch.

        Retorna True en tres casos:
        1. Si no hay suficientes hipótesis completadas aún (continuar buscando)
        2. Si early_stopping=True (confianza en que best_sum_logprobs es lo óptimo)
        3. Si ninguna hipótesis incompleta puede superar la peor almacenada

        Parameters
        ----------
        best_sum_logprobs : float
            Máxima suma de log-probs entre hipótesis incompletas activas
        cur_len : int
            Longitud actual de decodificación

        Returns
        -------
        bool
            True si la búsqueda puede terminar para este batch
        """
        # Si aún no hay num_beams completadas, necesita continuar buscando
        if len(self) < self.num_beams:
            return False
        
        # Si early_stopping habilitado, asume que best_sum_logprobs es lo mejor que se conseguirá
        elif self.early_stopping:
            return True
        
        # Compara el mejor score potencial (sin normalizar longitud) con el peor almacenado
        # Si even after normalizing por longitud futura, no puede mejorarse, termina
        else:
            # Normaliza el mejor score potencial por longitud actual
            cur_score = best_sum_logprobs / cur_len ** self.length_penalty
            # Retorna True si worst_score >= cur_score (ningún candidato puede superar)
            ret = self.worst_score >= cur_score
            return ret
