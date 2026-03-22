# 📊 PosFormer - Reconocimiento de Ecuaciones Matemáticas

**Trabajo Final de Ingeniería**

## 📝 Descripción

PosFormer es un modelo de Transformer para reconocer ecuaciones matemáticas manuscritas en imágenes y convertirlas a LaTeX.

El proyecto implementa una arquitectura dual-decoder que predice símbolos y sus posiciones en la imagen simultáneamente.

**Repositorio Original:** https://github.com/SJTU-DeepVisionLab/PosFormer

## 🏗️ Arquitectura

- **Encoder (DenseNet-16):** Extrae características de la imagen
- **Decoder:** Predice los símbolos de la ecuación
- **PosDecoder:** Predice posición y capa de cada símbolo
- **Beam Search:** Genera predicciones con múltiples hipótesis

## 📊 Parámetros

```yaml
Entrenamiento:
  - max_epochs: 300
  - learning_rate: 0.08
  - batch_size: 8
  - optimizer: SGD (momentum=0.9)

Modelo:
  - d_model: 256
  - num_layers: 16
  - nhead: 8
  - num_decoder_layers: 3
```

## 📁 Estructura

```
PosFormer/
├── train.py              # Script de entrenamiento
├── config.yaml           # Configuración
├── Pos_Former/
│   ├── model/            # Modelos (Encoder, Decoder)
│   ├── datamodule/       # Carga de datos
│   └── utils/            # Utilidades
├── data/                 # Datos de entrenamiento
└── lightning_logs/       # Checkpoints entrenados
```

## 🚀 Uso

### Instalar dependencias
```bash
pip install -r requirements.txt
```

### Entrenar
```bash
python train.py fit --config config.yaml
```

### Validar
```bash
python train.py validate --config config.yaml
```

### Testing
```bash
python train.py test --config config.yaml
```

## 🙏 Créditos

Basado en PosFormer de SJTU-DeepVisionLab - https://github.com/SJTU-DeepVisionLab/PosFormer

Ver [CREDITOS.md](CREDITOS.md) para información completa.

## 📄 Licencia

MIT License - Ver [LICENSE](LICENSE)
```
📂 PosFormer
   ├── 📦 data_crohme.zip
   ├── 📦 data_m2e.zip
   └── 📦 data_MNE.zip
```
The **MNE** dataset can now be downloaded [here](https://drive.google.com/file/d/1iiCxwt05v9a7jQIf074F1ltYLNxYe63b/view?usp=drive_link).

The **CROHME** dataset can be downloaded [CoMER/blob/master/data.zip](https://github.com/Green-Wood/CoMER/blob/master/data.zip) (provided by the **CoMER** project) 

The **M2E** dataset can be downloaded [here](https://www.modelscope.cn/datasets/Wente47/M2E/) 

If you have additional data, you can organize them in a similar way as shown below and compress them into a **zip** file (you can also modify the datamodule to directly input the files).
```
📂 data
   └── 📂 name_of_dataset
       ├── 📂 img
       │   ├── 0.png
       │   ├── 1.png
       │   └── ...
       └── caption.txt
```

### Checkpoint

[model weights](https://github.com/SJTU-DeepVisionLab/PosFormer/tree/main/lightning_logs/version_0/checkpoints)

### Training

For training, we utilize a single A800 GPU; however, an RTX 3090 GPU also provides sufficient memory to support a training batch size of 8. The training process is expected to take approximately 25 hours on a single A800 GPU.

```bash
cd PosFormer
python train.py --config config.yaml
```

### Evaluation 


```bash
cd PosFormer
# results will be printed in the screen and saved to lightning_logs/version_0 folder
bash eval_all_crohme.sh 0
```
### M2E Dataset
We provide source code files in ```m2e_pkg/``` for training and inferring the M2E dataset. You simply replace ```datamodule.py dictionary.txt label_make_multi.py vocab.py``` into ```Pos_Former/datamodule/``` , replace ```arm.py``` into ```Pos_Former/model/transformer/``` and run:

```bash
cd PosFormer
python train.py --config config_m2e.yaml

bash eval_all_m2e.sh 0
```
An integrated version of the code that eliminates the need for replacement files is coming soon
 ### TODO
 1. update LICENSE file 
 2. Improve README and samples

### Citation
```
@article{guan2024posformer,
  title={PosFormer: Recognizing Complex Handwritten Mathematical Expression with Position Forest Transformer},
  author={Guan, Tongkun and Lin, Chengyu and Shen, Wei and Yang, Xiaokang},
  journal={arXiv preprint arXiv:2407.07764},
  year={2024}
}
```

### License
```
- This code is only free for academic research purposes and licensed under the 2-clause BSD License. Parts of this project contain code from other sources, which are subject to their respective licenses.
```
