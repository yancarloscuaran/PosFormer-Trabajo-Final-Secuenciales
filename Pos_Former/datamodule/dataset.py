import torchvision.transforms as tr
from torch.utils.data.dataset import Dataset

from .transforms import ScaleAugmentation, ScaleToLimitRange

# Constantes utilizadas para las transformaciones de escala y tamaño de las imágenes.
K_MIN = 0.7  # Factor mínimo de escala para aumento de datos.
K_MAX = 1.4  # Factor máximo de escala para aumento de datos.

H_LO = 16  # Altura mínima permitida para las imágenes.
H_HI = 256  # Altura máxima permitida para las imágenes.
W_LO = 16  # Ancho mínimo permitido para las imágenes.
W_HI = 1024  # Ancho máximo permitido para las imágenes.

class CROHMEDataset(Dataset):
    """
    Dataset que recibe los datos preprocesados por datamodule.py y los entrega al DataLoader.

    Este dataset aplica transformaciones a las imágenes, como escalado y normalización, para
    garantizar que estén en el formato esperado por el modelo.
    """
    def __init__(self, ds, is_train: bool, scale_aug: bool) -> None:
        """
        Inicializa el dataset con los datos y las transformaciones necesarias.

        ds: Datos preprocesados (nombre de archivo, imagen, etiquetas).
        is_train: Indica si el dataset es para entrenamiento (True) o evaluación (False).
        scale_aug: Indica si se aplica aumento de escala a las imágenes.
        """
        super().__init__()
        self.ds = ds

        # Lista de transformaciones aplicadas a las imágenes.
        trans_list = []
        if is_train and scale_aug:
            # Aumento de escala para variar el tamaño de las imágenes durante el entrenamiento.
            trans_list.append(ScaleAugmentation(K_MIN, K_MAX))

        trans_list += [
            # Escala las imágenes para que estén dentro de un rango específico de tamaño.
            ScaleToLimitRange(w_lo=W_LO, w_hi=W_HI, h_lo=H_LO, h_hi=H_HI),
            tr.ToTensor(),  # Convierte las imágenes a tensores y las normaliza al rango [0, 1].
        ]
        self.transform = tr.Compose(trans_list)

    def __getitem__(self, idx):
        """
        Retorna un ejemplo del dataset con las transformaciones aplicadas.

        idx: Índice del ejemplo a retornar.
        """
        fname, img, caption = self.ds[idx]

        # Aplica las transformaciones a cada imagen del ejemplo.
        img = [self.transform(im) for im in img]

        return fname, img, caption

    def __len__(self):
        """
        Retorna el número total de ejemplos en el dataset.
        """
        return len(self.ds)
