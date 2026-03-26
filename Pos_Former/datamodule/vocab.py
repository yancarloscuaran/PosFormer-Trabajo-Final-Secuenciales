import os
from functools import lru_cache
from typing import Dict, List


@lru_cache()
def default_dict():
    """
    Retorna la ruta por defecto al archivo de diccionario que contiene los tokens LaTeX.
    """
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "dictionary.txt")


class CROHMEVocab:
    """
    Clase que define el vocabulario del modelo, incluyendo tokens especiales y métodos de conversión.

    El vocabulario contiene todos los tokens LaTeX que el modelo puede generar, incluyendo tokens especiales
    como <pad>, <sos> y <eos>.
    """

    # Índices reservados para tokens especiales.
    PAD_IDX = 0  # Token de padding, utilizado para rellenar secuencias.
    SOS_IDX = 1  # Token de inicio de secuencia.
    EOS_IDX = 2  # Token de fin de secuencia.

    def __init__(self, dict_path: str = default_dict()) -> None:
        """
        Inicializa el vocabulario cargando los tokens desde un archivo de diccionario.

        dict_path: Ruta al archivo que contiene los tokens LaTeX.
        """
        self.word2idx = dict()
        self.word2idx["<pad>"] = self.PAD_IDX
        self.word2idx["<sos>"] = self.SOS_IDX
        self.word2idx["<eos>"] = self.EOS_IDX

        # Carga los tokens del archivo de diccionario y asigna índices únicos.
        with open(dict_path, "r") as f:
            for line in f.readlines():
                w = line.strip()
                self.word2idx[w] = len(self.word2idx)

        # Crea un mapeo inverso de índices a palabras.
        self.idx2word: Dict[int, str] = {v: k for k, v in self.word2idx.items()}

    def words2indices(self, words: List[str]) -> List[int]:
        """
        Convierte una lista de palabras en una lista de índices según el vocabulario.

        Este método es utilizado en la función collate_fn de datamodule.py para preparar
        las secuencias de entrada al modelo.
        """
        return [self.word2idx[w] for w in words]

    def indices2words(self, id_list: List[int]) -> List[str]:
        """
        Convierte una lista de índices en una lista de palabras según el vocabulario.
        """
        return [self.idx2word[i] for i in id_list]

    def indices2label(self, id_list: List[int]) -> str:
        """
        Convierte una lista de índices en una cadena de texto LaTeX legible.

        Este método es utilizado en app.py para convertir las predicciones del modelo
        en texto LaTeX que puede ser mostrado al usuario.
        """
        words = self.indices2words(id_list)
        return " ".join(words)

    def __len__(self):
        """
        Retorna el tamaño del vocabulario.
        """
        return len(self.word2idx)


# Instancia global del vocabulario utilizada en todo el proyecto.
vocab = CROHMEVocab()
