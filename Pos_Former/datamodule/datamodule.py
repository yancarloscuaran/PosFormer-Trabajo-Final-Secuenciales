from dataclasses import dataclass
from typing import List, Optional, Tuple
from zipfile import ZipFile

import numpy as np
import pytorch_lightning as pl
import torch
from Pos_Former.datamodule.dataset import CROHMEDataset
from PIL import Image
from torch import FloatTensor, LongTensor
from torch.utils.data.dataloader import DataLoader

from .vocab import vocab
Data = List[Tuple[str, Image.Image, List[str]]]

# Tamaño máximo permitido para las imágenes en memoria, ajustable según la memoria de la GPU.
MAX_SIZE =32e4  # change here accroading to your GPU memory

def data_iterator(
    data: Data,
    batch_size: int,
    batch_Imagesize: int = MAX_SIZE,
    maxlen: int = 200,
    maxImagesize: int = 32e4,
):
    """
    Iterador que agrupa los datos en lotes basados en el tamaño de las imágenes y las secuencias.

    Los datos se ordenan por tamaño de imagen para optimizar la agrupación y evitar lotes desbalanceados.

    Parámetros:
    data: Lista de datos con nombre de archivo, imagen y etiquetas.
    batch_size: Número máximo de ejemplos por lote.
    batch_Imagesize: Tamaño máximo permitido para un lote en memoria.
    maxlen: Longitud máxima permitida para las secuencias de etiquetas.
    maxImagesize: Tamaño máximo permitido para una imagen individual.

    Retorna:
    Lista de lotes con nombres de archivo, características y etiquetas.
    """
    fname_batch = []
    feature_batch = []
    label_batch = []
    feature_total = []
    label_total = []
    fname_total = []
    biggest_image_size = 0

    # Ordena los datos por tamaño de imagen para optimizar la agrupación.
    data.sort(key=lambda x: x[1].size[0] * x[1].size[1])

    i = 0
    for fname, fea, lab in data:                  
        size = fea.size[0] * fea.size[1]
        fea = np.array(fea)           
        if size > biggest_image_size:
            biggest_image_size = size
        batch_image_size = biggest_image_size * (i + 1)
        if len(lab) > maxlen:
            print("sentence", i, "length bigger than", maxlen, "ignore")
        elif size > maxImagesize:
            print(
                f"image: {fname} size: {fea.shape[0]} x {fea.shape[1]} =  bigger than {maxImagesize}, ignore"
            )
        else:
            if batch_image_size > batch_Imagesize or i == batch_size:  # a batch is full
                fname_total.append(fname_batch)
                feature_total.append(feature_batch)
                label_total.append(label_batch)
                i = 0
                biggest_image_size = size
                fname_batch = []
                feature_batch = []
                label_batch = []
                fname_batch.append(fname)
                feature_batch.append(fea)
                label_batch.append(lab)
                i += 1
            else:
                fname_batch.append(fname)
                feature_batch.append(fea)
                label_batch.append(lab)
                i += 1

    # Agrega el último lote si quedó incompleto.
    fname_total.append(fname_batch)
    feature_total.append(feature_batch)
    label_total.append(label_batch)
    print("total ", len(feature_total), "batch data loaded")
    return list(zip(fname_total, feature_total, label_total))

def extract_data(archive: ZipFile, dir_name: str) -> Data:
    """
    Extrae las imágenes y fórmulas necesarias para un dataset desde un archivo zip.

    Primero se lee el archivo caption.txt para obtener las fórmulas asociadas a cada imagen.
    Luego, las imágenes se cargan una por una en memoria para evitar errores de puntero nulo.

    Parámetros:
    archive: Archivo zip que contiene los datos.
    dir_name: Nombre del directorio dentro del zip (por ejemplo: train, test_2014).

    Retorna:
    Lista de tuplas con nombre de imagen, imagen y fórmula asociada.
    """
    prefix = "data_MNE" if dir_name in ["N1", "N2", "N3"] else "data"
    with archive.open(f"{prefix}/{dir_name}/caption.txt", "r") as f:
        captions = f.readlines()
    data = []
    for line in captions:
        tmp = line.decode().strip().split()
        img_name = tmp[0]
        formula = tmp[1:]
        with archive.open(f"{prefix}/{dir_name}/img/{img_name}.bmp", "r") as f:
            img = Image.open(f).copy()
            data.append((img_name, img, formula))

    print(f"Extract data from: {dir_name}, with data size: {len(data)}")

    return data

@dataclass
class Batch:
    """
    Clase que encapsula un lote de datos preprocesados para el modelo.

    Contiene las imágenes, máscaras y secuencias de índices necesarias para el entrenamiento o evaluación.
    """
    img_bases: List[str]  # Nombres base de las imágenes en el lote.
    imgs: FloatTensor  # Tensor de imágenes [b, 1, H, W].
    mask: LongTensor  # Máscara binaria para las imágenes [b, H, W].
    indices: List[List[int]]  # Índices de las secuencias de etiquetas [b, l].

    def __len__(self) -> int:
        return len(self.img_bases)

    def to(self, device) -> "Batch":       
        # Mueve los tensores del lote al dispositivo especificado (CPU o GPU).
        return Batch(
            img_bases=self.img_bases,
            imgs=self.imgs.to(device),
            mask=self.mask.to(device),
            indices=self.indices,
        )

def collate_fn(batch):
    """
    Función de colación para combinar ejemplos individuales en un lote.

    Asegura que todas las imágenes tengan el mismo tamaño rellenando con ceros.
    """
    assert len(batch) == 1
    batch = batch[0]
    fnames = batch[0]
    images_x = batch[1]
    seqs_y = [vocab.words2indices(x) for x in batch[2]]

    heights_x = [s.size(1) for s in images_x]
    widths_x = [s.size(2) for s in images_x]

    n_samples = len(heights_x)
    max_height_x = max(heights_x)
    max_width_x = max(widths_x)

    x = torch.zeros(n_samples, 1, max_height_x, max_width_x)
    x_mask = torch.ones(n_samples, max_height_x, max_width_x, dtype=torch.bool)
    for idx, s_x in enumerate(images_x):
        x[idx, :, : heights_x[idx], : widths_x[idx]] = s_x
        x_mask[idx, : heights_x[idx], : widths_x[idx]] = 0

    return Batch(fnames, x, x_mask, seqs_y)

def build_dataset(archive, folder: str, batch_size: int):
    """
    Construye un dataset a partir de un archivo zip y un directorio específico.

    Llama a extract_data para obtener los datos y luego utiliza data_iterator para agruparlos en lotes.
    """
    data = extract_data(archive, folder)
    return data_iterator(data, batch_size)

class CROHMEDatamodule(pl.LightningDataModule):
    """
    Módulo de datos para manejar la carga y preprocesamiento de los datasets CROHME y MNE.

    Este módulo utiliza PyTorch Lightning para organizar los datasets de entrenamiento, validación y prueba.
    """
    def __init__(
        self,
        zipfile_path: str = "data_crohme.zip",
        test_year: str = "2014",
        train_batch_size: int = 24,
        eval_batch_size: int = 12,
        num_workers: int = 5,
        scale_aug: bool = True,
    ) -> None:
        """
        Inicializa el módulo de datos con los parámetros necesarios.

        zipfile_path: Ruta al archivo zip que contiene los datos.
        test_year: Año del conjunto de prueba (por defecto 2014).
        train_batch_size: Tamaño de lote para entrenamiento (24 por defecto para un balance entre memoria y rendimiento).
        eval_batch_size: Tamaño de lote para evaluación (12 por defecto para reducir el uso de memoria).
        num_workers: Número de procesos para cargar datos en paralelo.
        scale_aug: Indica si se aplica aumento de escala a las imágenes.
        """
        super().__init__()
        assert isinstance(test_year, str)
        self.zipfile_path = zipfile_path
        self.test_year = test_year
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.scale_aug = scale_aug
        print(f"Load data from: {self.zipfile_path}")

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Configura los datasets para las etapas de entrenamiento, validación y prueba.

        Durante la etapa "fit", se crean los datasets de entrenamiento y validación.
        Durante la etapa "test", se crea únicamente el dataset de prueba.
        """
        with ZipFile(self.zipfile_path) as archive:
            if stage == "fit" or stage is None:
                self.train_dataset = CROHMEDataset(
                    build_dataset(archive, "train", self.train_batch_size),
                    True,
                    self.scale_aug,
                )
                self.val_dataset = CROHMEDataset(
                    build_dataset(archive, self.test_year, self.eval_batch_size),
                    False,
                    self.scale_aug,
                )
            if stage == "test" or stage is None:
                self.test_dataset = CROHMEDataset(
                    build_dataset(archive, self.test_year, self.eval_batch_size),
                    False,
                    self.scale_aug,
                )

    def train_dataloader(self):
        """
        Retorna el DataLoader para el conjunto de entrenamiento.
        """
        return DataLoader(
            self.train_dataset,
            shuffle=True,             
            num_workers=self.num_workers,    
            collate_fn=collate_fn,     
        )

    def val_dataloader(self):
        """
        Retorna el DataLoader para el conjunto de validación.
        """
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

    def test_dataloader(self):
        """
        Retorna el DataLoader para el conjunto de prueba.
        """
        return DataLoader(
            self.test_dataset,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )
