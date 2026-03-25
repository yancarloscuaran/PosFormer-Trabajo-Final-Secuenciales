# 📊 PosFormer - Reconocimiento de Ecuaciones Matemáticas

## 📑 Resumen (Abstract)

Este trabajo describe la implementación y el proceso de inferencia de PosFormer (Position Forest Transformer), una arquitectura encoder-decoder orientada al reconocimiento de expresiones matemáticas manuscritas (HMER), fundamentada en la propuesta original de Guan et al. (2024). El modelo optimiza la transcripción de imágenes de ecuaciones a secuencias LaTeX mediante la integración de un Position Forest, diseñado para capturar de forma robusta las complejas relaciones jerárquicas y espaciales intrínsecas a la notación matemática, junto con un módulo de Corrección de Atención Implícita (IAC) que refina la precisión del mecanismo de atención durante la decodificación. A través de este enfoque, se aborda la naturaleza bidimensional y estructural de los símbolos manuscritos, mejorando la fidelidad y el rendimiento en tareas de recuperación de secuencias técnicas.

Implementamos el proceso de inferencia del modelo utilizando los pesos preentrenados proporcionados por los autores, y desarrollamos una interfaz interactiva con Streamlit que permite cargar imágenes de expresiones matemáticas y visualizar el resultado en LaTeX en tiempo real. Los resultados obtenidos sobre los datasets CROHME 2014, 2016 y 2019 alcanzaron un ExpRate de 62.68%, 61.03% y 64.97% respectivamente. Adicionalmente, sobre expresiones multilínea (M2E) el modelo obtuvo un ExpRate de 58.33%, y sobre expresiones anidadas complejas (MNE) alcanzó ganancias de hasta 10.04% sobre el estado del arte en expresiones de mayor complejidad. Estos resultados validan el correcto funcionamiento del modelo con los pesos preentrenados.

## 📝 Introducción

El reconocimiento de expresiones matemáticas manuscritas (*HMER, por sus siglas en inglés*) es una tarea que busca convertir imágenes de ecuaciones escritas a mano en secuencias LaTeX interpretables por computador. Esta tarea tiene aplicaciones directas en educación digital, digitalización de documentos científicos y sistemas de corrección automática, contextos donde la interacción entre humanos y máquinas a través de notación matemática es cada vez más relevante.

El principal desafío de HMER radica en dos factores: la complejidad de las relaciones entre símbolos matemáticos, que incluyen estructuras anidadas como fracciones dentro de exponentes o radicales dentro de sumatorias, y la diversidad de estilos de escritura a mano, que genera variaciones significativas en escala y forma. Existen métodos que modelan la expresión matemática como una **_estructura de árbol_** represnetando la relaciones entre los simbolos a través de una tupla de tres valores (padre, hijo, relación padre-hijo), para luego decodifircarlo en una secuencia de LaTeX. Por otro lado, los **_métodos basados en secuencia_** modelan el reconocimiento de la expresión como una tarea de extremo a extremo de imagen a secuencia LaTeX y usa una arquitectura de atención encoder-decoder para predecir los simbolos de forma autoregresiva. Estos métodos tienen problemas para procesar expresiones complejas y anidadas. Para abordar este problema, los autores proponen PosFormer. Este modelo:

- Considera las relaciones entre símbolos para permitir el reconocimiento de expresiones matemáticas complejas. Para ello, se codifica la secuencia de expresiones matemáticas de LaTeX como una estructura de bosque de posiciones, a cada símbolo se asigna un identificador de posición espacial relativa en una imagen bidimensional.
- Con esta informacion el entrenamiento analiza las relaciones, niveles y anidaciones para aprender una representacion de los niveles de jerarquia de las relaciones y sus posiciones.
- Introduce un módulo de correccion de la atención en la arquitectura del decoder.

PosFormer incorpora explícitamente la comprensión de las relaciones posicionales entre símbolos dentro de una arquitectura Transformer encoder-decoder. A diferencia del decoder tradicional que solo genera la secuencia LaTeX, PosFormer añade una tarea auxiliar de reconocimiento de posición durante el entrenamiento, lo que le permite aprender representaciones más ricas sin costo adicional en inferencia.

En este trabajo implementamos el proceso de inferencia usando los pesos preentrenados provistos por los autores, evidenciamos su funcionamiento sobre los datasets de evaluación CROHME, M2E y MNE, y resultados funcionales mediante una interfaz interactiva desarrollada con Streamlit. 

El proyecto implementa una arquitectura dual-decoder que predice símbolos y sus posiciones en la imagen simultáneamente. Este es un ejericicio de procesamiento de datos secuenciales donde la entrada es una imagen y la salida es una secuencia de tokens LaTeX generada de forma autoregresiva

**Referencia:** 

Guan, T., Lin, C., Shen, W., & Yang, X. (2024, September). Posformer: recognizing complex handwritten mathematical expression with position forest transformer. In European Conference on Computer Vision (pp. 130-147). Cham: Springer Nature Switzerland. From:https://doi.org/10.48550/arXiv.2407.07764.

**Repositorio Original:** https://github.com/SJTU-DeepVisionLab/PosFormer

## 🏗️ Generalidades de la Arquitectura

- **Encoder (DenseNet-16):** Extrae características de la imagen.
- **Decoder:** Predice los símbolos de la ecuación.
- **PosDecoder:** Predice posición y nivel de anidación de cada símbolo.
- **Beam Search:** Genera predicciones con múltiples hipótesis,de forma autorregresiva.

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

Ver [CREDITOS.md](CREDITOS.md) para información completa.

## 📄 Licencia

MIT License - Ver [LICENSE](LICENSE)
```
- This code is only free for academic research purposes and licensed under the 2-clause BSD License. Parts of this project contain code from other sources, which are subject to their respective licenses.
```
=======
# Marco Teórico: Explicación Detallada de la Arquitectura PosFormer

<img width="1969" height="1103" alt="image" src="https://github.com/user-attachments/assets/247cebae-6366-4a5a-9a27-8b85a803c52d" />

Fuente: Elaboración propia.


$\color{#4682B4}{\textbf{1. Preprocesamiento de Datos}}$

El flujo de preparación de datos comienza con el **Scale Augmentation**, esto fuerza al modelo a ser invariante al tamaño de la escritura, seguido de la conversión a tensores y normalización de las imágenes para estabilizar los gradientes durante el entrenamiento. Posteriormente, se aplica **zero-padding** para uniformizar las dimensiones espaciales, generando simultáneamente la `img_mask` que permite al cross-attention del decoder ignorar el fondo, mientras que el data_iterator optimiza el uso de memoria agrupando las imágenes dinámicamente por volumen de píxeles en lugar de usar un tamaño de batch fijo. Finalmente. Cada símbolo LaTeX del vocabulario se mapea a un índice entero único, con índices reservados para <pad>, <sos> y <eos>, posteriormente las secuencias se procesan mediante codificación bidireccional (L2R y R2L), lo que duplica el batch efectivo y mejora drásticamente la robustez de la inferencia al permitir que el beam search combine ambas direcciones de lectura.


