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

<img width="2000" height="1143" alt="image" src="https://github.com/user-attachments/assets/65262cde-df73-48ba-9d7a-198027484863" />


Fuente: Elaboración propia.


### $\color{#4682B4}{\textbf{1. Preprocesamiento de Datos}}$

El flujo de preparación de datos comienza con el **Scale Augmentation**, esto fuerza al modelo a ser invariante al tamaño de la escritura, seguido de la conversión a tensores y normalización de las imágenes para estabilizar los gradientes durante el entrenamiento. Posteriormente, se aplica **zero-padding** para uniformizar las dimensiones espaciales, generando simultáneamente la `img_mask` que permite al cross-attention del decoder ignorar el fondo, mientras que el data_iterator optimiza el uso de memoria agrupando las imágenes dinámicamente por volumen de píxeles en lugar de usar un tamaño de batch fijo. Finalmente. Cada símbolo LaTeX del vocabulario se mapea a un índice entero único, con índices reservados para <pad>, <sos> y <eos>, posteriormente las secuencias se procesan mediante codificación bidireccional (L2R y R2L), lo que duplica el batch efectivo y mejora drásticamente la robustez de la inferencia al permitir que el beam search combine ambas direcciones de lectura.

### $\color{#4682B4}{\textbf{2. Encoder: DenseNet + Proyección Visual}}$

El Encoder de la arquitectura utiliza una DenseNet como backbone fundamental. En esta estructura, cada capa recibe como entrada la concatenación de todas las capas precedentes, lo que garantiza un flujo directo de gradientes y evita su desvanecimiento. Este diseño permite preservar y acumular características espaciales a múltiples escalas: desde bordes y líneas básicas en las primeras capas, hasta estructuras semánticas complejas en las capas más profundas.

**Arquitectura DenseNet:** La extracción de características sigue un proceso de reducción de resolución y profundización jerárquica:

- **Conv2d 7×7 stride=2 + BatchNorm + ReLU:** Genera una reducción inicial de resolución a** $H/2 \times W/2$** para capturar la estructura global del trazo.
- **MaxPool 2×2:** Realiza una segunda reducción agresiva hasta **$H/4 \times W/4$**, descartando variaciones finas de posición que podrían generar ruido visual.
- **DenseBlock1 + Transition1 (avg_pool)**: Extrae características locales y comprime los canales por la mitad, reduciendo la resolución a **$H/8 \times W/8$**.
- **DenseBlock2 + Transition2 (avg_pool):** Aumenta la riqueza semántica y lleva la resolución espacial a su valor final de **$H/16 \times W/16$**.
- **DenseBlock3:** Profundiza la representación final sin aplicar downsampling adicional, manteniendo la resolución en $H/16 \times W/16$.
- **Bottleneck Layers:** Implementación interna de capas **BN → Conv1×1 → BN → Conv3×3**. La convolución 1×1 reduce canales antes de la de 3×3, optimizando el costo computacional.
- **Dropout y Concatenación:** Se aplica **Dropout (p=0.2)** para evitar la co-adaptación de características, mientras que la concatenación constante preserva la identidad de cada feature a lo largo de toda la red.

**Proyección de canales: Conv2d 1×1:** Este paso disminuye los canales de salida de la DenseNet a una dimensión fija de **$d\_model = 256$**. Este paso establece el espacio de representación vectorial que esperan los decoders, transformando los mapas de activación en embeddings visuales compatibles con el Transformer.

**Codificación Posicional 2D Sinusoidal (`ImgPosEnc`):** Se adicionan coordenadas espaciales directamente en los features 2D generados por el encoder. Cada posición espacial recibe un encoding independiente por eje, permitiendo que el modelo razone sobre desplazamientos horizontales y verticales por separado. Estas 

**Normalización Acumulativa:** Las coordenadas se escalan relativamente al tamaño real de cada imagen (ignorando el padding), asegurando que una posición relativa tenga el mismo encoding sin importar el tamaño original de la muestra.

**LayerNorm Final (Normalización K, V):** Se aplica una **LayerNorm** sobre los features visuales tras añadir el encoding posicional. Esto estabiliza la distribución de valores que recibirán los decoders como "memoria" durante el proceso de *cross-attention*.

**Salida del Encoder:** `Mapa de Features` y `Memory Mask. El resultado final mantiene la estructura espacial 2D intacta. Cada posición $(h, w)$ es un vector de **256 dimensiones** que representa una región de la imagen original reducida 16 veces en cada eje. La **Memory Mask** es una máscara de padding que fuerza las posiciones vacías a $-\infty$ antes del cálculo del softmax en el decodificador, garantizando que el modelo ignore completamente las zonas sin información visual.

### $\color{#4682B4}{\textbf{3. Codificación del Bosque de Posiciones (Position Forest)}}$

El Position Forest transforma la secuencia LaTeX en etiquetas de posición espacial por símbolo, tinedo en cuenta subestructuras y roles de posición. Para la construcción del bósque de posiciones un algoritmo asigna identificadores **M (Middle/Root)** al cuerpo principal o raíz de la expresion, y teniendo en cuenta las estructura precedentes **L (Left/Upper)** a los simbolos que se ubican e Parte superior (ej. numerador o exponente) y ** R (Right/Lower)** a los que se ubican en la parte inferior (ej. denominador o subíndice). Para posiciones anidadas, los indentificadores se concatenan para reflejar la jerárquica.

Dado que los identificadores varían en longitud según la profundidad de anidación, se truncan o rellenan con ceros (pad) a longitud fija 5 para poder apilarlos en una matriz uniforme. Aunque el artículo menciona que el nivel máximo de anidamiento es 3 (lo que requeriría 4 posiciones: Raíz + 3 niveles), el código asegura un margen de seguridad de 5 para la matriz. Esta matriz es la entrada al **PosDecoder Cross Attention**. Cada fila se proyecta a `d_model=256` vía una capa no lineal que incluye una proyección lineal **GeLu** - **ξ(Linear→GeLU→LayerNorm)**.

Los identificadores (ground truth) se descompone en dos señales de supervisión independientes: el **nivel de anidación**, definido por la longitud del identificador menos uno, y **la posición relativa**, extraída de su penúltimo carácter. Así, el PosDecoder aprende por separado tanto la profundidad jerárquica como la ubicación espacial de cada símbolo.

### $\color{#4682B4}{\textbf{4. Decodificador (Decoder)}}$

El Decoder genera la secuencia LaTeX token a token, actuando como el único componente activo durante la inferencia para traducir el mapa visual del encoder en texto LaTeX. Inicialmente, los índices del vocabulario se transforman en vectores densos de 256 dimensiones mediante una capa de `nn.Embedding` estabilizada con LayerNorm. A estos vectores se les suma una codificación posicional sinusoidal absoluta (WordPosEnc), la cual inyecta información sobre el orden secuencial para que el mecanismo de auto-atención pueda distinguir la cronología de los tokens generados.

La arquitectura del Decoder consta de tres capas a su vez compuestas por subcapas multi-head de self-attention, cross-attention y redes Feedforward (FFN). El mecanismo **self-attention** de ocho cabezas implementa una doble máscara: una causal de tipo triangular superior para bloquear el acceso a tokens futuros `causal_mask` y una máscara de padding para ignorar posiciones sin contenido `tgt_pad_mask`. Tras cada operación, las conexiones residuales y la normalización de capa (LayerNorm) garantizan un flujo de gradiente estable y la consistencia en las activaciones de la red.

El elemento distintivo de este módulo es la **cross-attention** equipada con **Implicit Attention Correction (IAC)**. El IAC aborda una limitación crítica en modelos como CoMER, donde los símbolos estructurales (como ^, _, {) carecen de un referente visual directo, lo que suele corromper el mapa de cobertura al distribuir atención difusa sobre regiones ya procesadas. Mediante una función indicadora, el IAC anula las contribuciones de estos símbolos antes de su acumulación. Esta cobertura corregida se procesa mediante una convolución y se resta a la atención actual en las capas 2 y 3, optimizando la precisión de la lectura visual.

Finalmente, el flujo de datos atraviesa una **Feed-Forward Network (FFN)** que expande la dimensionalidad interna a **1024** para introducir capacidades no lineales mediante activaciones ReLU y Dropout. El proceso concluye con una proyección lineal hacia el tamaño total del vocabulario, produciendo los logits que definen la distribución de probabilidad para el siguiente símbolo LaTeX. Esta distribución es procesada por un **Softmax**, ya sea de forma implícita durante el cálculo de la pérdida en el entrenamiento o explícitamente durante el proceso de beam search en la fase de inferencia.

### $\color{#4682B4}{\textbf{5. Decodificardor Posicional (PosDecoder)}}$

El PosDecoder actúa como un decodificador auxiliar diseñado para aprender las relaciones posicionales del bosque jerárquico exclusivamente durante la fase de entrenamiento. El proceso se inicia con la proyección de los identificadores de posición de cinco dimensiones al espacio de representación de 256 dimensiones mediante una transformación **$\xi$ (Linear $\to$ GeLU $\to$ LayerNorm)**. Se usa la activación **GeLU** en lugar de **ReLU** debido a su capacidad para preservar activaciones negativas pequeñas, lo cual resulta fundamental para distinguir matices estructurales entre identificadores similares. A esta representación se le suma una **codificación sinusoidal absoluta (WordPosEnc)** para que el mecanismo de atención pueda diferenciar el orden secuencial en la generación de los identificadores. 

La arquitectura del PosDecoder es similar que la del Decoder, sin embargo a diferencia del Decoder de reconocimiento, el PosDecoder utiliza como **consulta ($Q$), clave ($K$) y valor ($V$)** los **identificadores proyectados**, permitiendo que cada posición de la secuencia atienda a su propio contexto estructural de forma autorregresiva. Tras cada operación, las conexiones residuales y la normalización de capa garantizan la estabilidad del gradiente y la consistencia de las activaciones.

Este componente es que se elimina por completo durante la fase de inferencia. Dado que su único propósito es proveer una señal de supervisión posicional que el Decoder internaliza durante el entrenamiento conjunto, su remoción en producción permite ahorrar aproximadamente 3 millones de parámetros. Esto optimiza el costo computacional en entornos de despliegue sin sacrificar la robustez estructural que el modelo ha adquirido.Finalmente, el flujo de datos culmina en dos cabezas de predicción especializadas que supervisan el aprendizaje jerárquico. La primera, denominada `layernum_proj`, predice la profundidad del símbolo en el árbol de la expresión (niveles 0 a 4), forzando al modelo a entender el grado de anidación. La segunda, `pos_proj`, identifica el rol espacial inmediato del símbolo, distinguiendo si se trata de un numerador, denominador o parte del cuerpo principal y asegurando que el sistema capture con precisión la topología tridimensional de la fórmula matemática.

### $\color{#4682B4}{\textbf{6. Función de Pérdida}}$

El entrenamiento utiliza una **Multi-task Loss** que combina la **Cross-Entropy** de la secuencia LaTeX con las señales de anidación y posición relativa del PosDecoder. Para evitar distorsiones en el gradiente, se aplican padding masks que aseguran que el modelo ignore los tokens no válidos en el cálculo del error. Los términos auxiliares se ponderan con un factor de 0.25 y la suma total se normaliza por 1.5, garantizando que el reconocimiento de texto permanezca como la tarea prioritaria.

$$L_{total} = \frac{L_{seq} + 0.25 \cdot L_{layer} + 0.25 \cdot L_{pos}}{1.5}$$

### $\color{#4682B4}{\textbf{7. Inferencia — Beam Search}}$

En validación/test no se usa el PosDecoder. El encoder procesa la imagen **feature [1, h, w, 256]**. **Beam search** inicia con `<sos>` y en cada paso ejecuta el Decoder completo para obtener `logits → log_softmax → topk(10)` selecciona los 10 tokens más probables. Mantiene 10 caminos activos, sumando log-probabilidades acumuladas. Repite hasta `<eos>` o `max_len=200`. Selecciona la secuencia con mayor probabilidad acumulada total y entreag la secuencia LaTeX final.
