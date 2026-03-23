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

Guan, T., Lin, C., Shen, W., & Yang, X. (n.d.). PosFormer: Recognizing Complex Handwritten Mathematical Expression with Position Forest Transformer. Retrieved March 14, 2026, from https://github.com/SJTU-DeepVisionLab/PosFormer 

Ver [CREDITOS.md](CREDITOS.md) para información completa.

## 📄 Licencia

MIT License - Ver [LICENSE](LICENSE)
```
- This code is only free for academic research purposes and licensed under the 2-clause BSD License. Parts of this project contain code from other sources, which are subject to their respective licenses.
```
=======
# PosFormer-Trabajo-Final-Secuenciales
Analisis de Arquitectura.
>>>>>>> d8dc53fddcd3bfeac086dea054d128c8607a7083
