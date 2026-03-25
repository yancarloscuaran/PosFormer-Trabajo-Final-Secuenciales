[Volver](README.md)

# Metodología

## 1. Proceso de implementación: 

El objetivo de esta sección es describir cómo implementamos el proceso de inferencia de PosFormer sobre una máquina local con Windows, usando los pesos preentrenados provistos por los autores del artículo.

El punto de partida fue el repositorio oficial del modelo disponible en GitHub. Dado que el entorno de desarrollo original fue diseñado para Linux con GPU NVIDIA de alta capacidad, fue necesario realizar una serie de adaptaciones para ejecutarlo correctamente en Windows con CPU.

## 2. Herramientas utilizadas: 

| Herramienta | Versión | Rol |
| :--- | :--- | :--- |
| Python | 3.7 | Lenguaje base del proyecto |
| PyTorch | 1.8.1 + cpu | Framework de deep learning |
| PyTorch Lightning | 1.4.9 | Gestión del ciclo de entrenamiento e inferencia |
| torchmetrics | 0.6.0 | Métricas de evaluación |
| Streamlit | latest | Interfaz interactiva de inferencia |
| Miniconda | latest | Gestión del entorno virtual |
| VS Code | latest | Entorno de desarrollo |

## 3. Uso de los pesos preentrenados

Los autores del artículo proveen los pesos preentrenados directamente en el repositorio de GitHub, ubicados en:

```bash
lightning_logs/version_0/checkpoints/best.ckpt  (75 MB)
```

Estos pesos fueron entrenados por los autores sobre el dataset CROHME usando una GPU NVIDIA A800. Para cargarlos en inferencia utilizamos el método load_from_checkpoint de PyTorch Lightning:

```bash
model = LitPosFormer.load_from_checkpoint(
    "lightning_logs/version_0/checkpoints/best.ckpt",
    map_location="cpu"
)
model.eval()
```

El parámetro map_location="cpu" fue necesario porque los pesos fueron entrenados en GPU pero nuestra máquina ejecuta la inferencia en CPU.

## 4. Adaptaciones realizadas

Durante la implementación encontramos las siguientes incompatibilidades que requirieron ajustes:

PyTorch con CUDA: la versión cudatoolkit = 11.1 requerida por el proyecto no estaba disponible en los canales de conda para Windows. Se instaló cudatoolkit = 11.3 y luego se reinstalió PyTorch vía pip en su versión CPU para evitar un error de DLL (caffe2_detectron_ops.dll).

Scripts de evaluación en bash: los scripts eval_all_crohme.sh y eval_all_mne.sh usan comandos de Linux (lgeval, bash) que no funcionan en Windows. Se adaptó la evaluación ejecutando directamente el script Python interno:

```bash
python scripts/test/test.py 0 2014
```

Evaluación de MNE: el script original asumía que todos los datasets estaban dentro de data_crohme.zip. Se modificó scripts/test/test.py y Pos_Former/datamodule/datamodule.py para detectar automáticamente el archivo zip correcto según el dataset:
 
```bash
zip_file = "data_MNE.zip" if test_year in ["N1", "N2", "N3"] else "data_crohme.zip"
```

Trainer sin GPU: el script de evaluación tenía gpus=1 configurado. Se cambió a gpus=0, accelerator='cpu' para ejecutar en CPU.

## 5. Interfaz interactiva con Streamlit
Desarrollamos una interfaz con Streamlit que permite cargar una imagen de expresión matemática manuscrita y obtener el código LaTeX generado por el modelo en tiempo real. Durante las pruebas iniciales identificamos que el modelo fue entrenado con imágenes de fondo negro y trazos blancos, lo que causaba predicciones incorrectas con imágenes de fondo claro. Para resolver esto, la interfaz aplica automáticamente una binarización a la imagen de entrada que normaliza cualquier imagen al formato esperado por el modelo independientemente de su fondo original.

# Desarrollo e implementación
Pasos para ejecutar el proyecto, explicación de cómo se cargan los pesos y explicación del proceso de preprocesamiento e inferencia.

## 1. Requisitos previos 
Para ejecutar el proyecto se necesita tener instalado Git, Miniconda y VSCode. El proyecto fue probado en Windows 11 con una GPU NVIDIA RTX 3050 (4GB VRAM), aunque la inferencia corre en CPU.

## 2. Instalación 
```bash
# Clonar el repositorio
git clone https://github.com/SJTU-DeepVisionLab/PosFormer.git
cd PosFormer

# Crear y activar el entorno
conda create -n PosFormer python=3.7
conda activate PosFormer

# Instalar dependencias
conda install pytorch=1.8.1 torchvision=0.9.1 cudatoolkit=11.3 -c pytorch -c conda-forge
pip uninstall torch torchvision -y
pip install torch==1.8.1+cpu torchvision==0.9.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
conda install pytorch-lightning=1.4.9 torchmetrics=0.6.0 -c conda-forge
conda install pandoc=1.19.2.1 -c conda-forge
pip install pillow==9.3.0 opencv-python einops typer gdown streamlit

# Dataset CROHME
gdown --fuzzy "https://github.com/Green-Wood/CoMER/raw/master/data.zip" -O data_crohme.zip
Expand-Archive -Path data_crohme.zip -DestinationPath .

# Dataset MNE (incluido en el repositorio)
Expand-Archive -Path data_MNE.zip -DestinationPath .
```
Los pesos preentrenados ya vienen incluidos en el repositorio clonado en lightning_logs/version_0/checkpoints/best.ckpt.

## 3. Proceso de inferencia

Para evaluar sobre los datasets:

```bash
# CROHME
python scripts/test/test.py 0 2014
python scripts/test/test.py 0 2019
# MNE
python scripts/test/test.py 0 N1
python scripts/test/test.py 0 N2
python scripts/test/test.py 0 N3
```

## 4. Interfaz Streamlit

Para demostrar la inferencia de forma interactiva:
```bash
streamlit run app.py
```


