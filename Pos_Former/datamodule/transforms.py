from typing import List

import cv2
import numpy as np
import torch
from torch import Tensor


class ScaleToLimitRange:
    """Escala la imagen para mantenerla dentro de rangos de altura y ancho permitidos.

    Esta transformación es aplicada por dataset.py durante la carga de datos. Ajusta el tamaño
    de la imagen respetando su proporción de aspecto (aspect ratio) para que encaje dentro del rango
    especificado [w_lo, w_hi] x [h_lo, h_hi]. Si la imagen es demasiado grande, la reduce; si es
    muy pequeña, la amplía. Esto normaliza el tamaño de las imágenes de entrada al modelo.
    """
    def __init__(self, w_lo: int, w_hi: int, h_lo: int, h_hi: int) -> None:
        assert w_lo <= w_hi and h_lo <= h_hi
        self.w_lo = w_lo
        self.w_hi = w_hi
        self.h_lo = h_lo
        self.h_hi = h_hi

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """Aplica el escalado manteniendo la proporción de aspecto.

        Parameters
        ----------
        img : np.ndarray
            Imagen en formato numpy (H, W) o (H, W, C)

        Returns
        -------
        np.ndarray
            Imagen escalada dentro del rango permitido
        """
        h, w = img.shape[:2]
        r = h / w
        # Calcula los límites de proporción de aspecto permitida según los rangos
        lo_r = self.h_lo / self.w_hi
        hi_r = self.h_hi / self.w_lo
        assert lo_r <= h / w <= hi_r, f"img ratio h:w {r} not in range [{lo_r}, {hi_r}]"

        # Intenta reducir si alguna dimensión excede el máximo permitido
        scale_r = min(self.h_hi / h, self.w_hi / w)
        if scale_r < 1.0:
            # La imagen es demasiado grande (altura o ancho mayor que los límites), reduce
            img = cv2.resize(
                img, None, fx=scale_r, fy=scale_r, interpolation=cv2.INTER_LINEAR
            )
            return img

        # Intenta ampliar si alguna dimensión es menor que el mínimo permitido
        scale_r = max(self.h_lo / h, self.w_lo / w)
        if scale_r > 1.0:
            # La imagen es demasiado pequeña (altura o ancho menor que los límites), amplía
            img = cv2.resize(
                img, None, fx=scale_r, fy=scale_r, interpolation=cv2.INTER_LINEAR
            )
            return img

        # La imagen ya está dentro del rango permitido, no requiere escalado
        assert self.h_lo <= h <= self.h_hi and self.w_lo <= w <= self.w_hi
        return img


class ScaleAugmentation:
    """Aplica aumento de datos mediante escalado aleatorio de la imagen.

    Esta transformación es aplicada por dataset.py solo durante el entrenamiento (is_train=True)
    para incrementar la variabilidad de los datos de entrenamiento. Escala la imagen con un factor
    aleatorio entre lo e hi, lo que simula imágenes de diferentes tamaños en el conjunto de datos.
    Esto mejora la generalización del modelo al permitirle aprender con variaciones de escala.
    """
    def __init__(self, lo: float, hi: float) -> None:
        assert lo <= hi
        self.lo = lo
        self.hi = hi

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """Escala la imagen con un factor aleatorio.

        Parameters
        ----------
        img : np.ndarray
            Imagen en formato numpy (H, W) o (H, W, C)

        Returns
        -------
        np.ndarray
            Imagen escalada por un factor k uniformemente aleatorio en [lo, hi]
        """
        # Genera factor de escala aleatorio entre lo (0.7) e hi (1.4) para data augmentation
        k = np.random.uniform(self.lo, self.hi)
        img = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_LINEAR)
        return img
