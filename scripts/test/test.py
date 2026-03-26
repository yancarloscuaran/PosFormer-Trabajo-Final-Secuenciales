import os
import sys
sys.path.append("/path/to/PosFormer")
import typer
import zipfile
from Pos_Former.datamodule import CROHMEDatamodule
from Pos_Former.lit_posformer import LitPosFormer
from pytorch_lightning import Trainer, seed_everything

# Fija la semilla para garantizar reproducibilidad en los resultados
seed_everything(7)

def cal_distance(word1, word2):
    """
    Calcula la distancia de edición (Levenshtein) entre dos palabras.
    Esto se utiliza para medir la similitud entre la predicción del modelo y la referencia.
    """
    m = len(word1)
    n = len(word2)
    if m*n == 0:
        return m+n
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range (m+1):
        dp[i][0] = i
    for j in range(n+1):
        dp[0][j] = j
    for i in range(1, m+1):
        for j in range(1, n+1):
            a = dp[i-1][j] + 1
            b = dp[i][j-1] + 1
            c = dp[i-1][j-1]
            if word1[i-1] != word2[j-1]:
                c += 1
            dp[i][j] = min(a, b, c)
    return dp[m][n]

def main(version: str, test_year: str):
    """
    Función principal para evaluar el modelo PosFormer.
    Carga el modelo entrenado, selecciona el dataset de prueba y calcula las métricas de evaluación.
    """
    # Ruta al checkpoint del modelo entrenado
    ckp_folder = os.path.join("lightning_logs", f"version_{version}", "checkpoints")
    fnames = os.listdir(ckp_folder)
    assert len(fnames) == 1  # Asegura que solo haya un checkpoint en la carpeta
    ckp_path = os.path.join(ckp_folder, fnames[0])
    print(f"Test con el archivo: {fnames[0]}")

    # Inicializa el entrenador de PyTorch Lightning para realizar la evaluación
    trainer = Trainer(logger=False, gpus=0, accelerator='cpu')

    # Selección del dataset según el año de prueba
    zip_file = "data_MNE.zip" if test_year in ["N1", "N2", "N3"] else "data_crohme.zip"
    dm = CROHMEDatamodule(zipfile_path=zip_file, test_year=test_year, eval_batch_size=1)

    # Carga el modelo desde el checkpoint
    model = LitPosFormer.load_from_checkpoint(ckp_path)

    # Realiza la evaluación del modelo sobre el dataset seleccionado
    trainer.test(model, datamodule=dm)

    # Diccionario para almacenar las referencias de las expresiones matemáticas
    caption = {}

    # Carga las referencias de las expresiones matemáticas desde el archivo de captions
    with zipfile.ZipFile(zip_file) as archive:
        folder = "data_MNE" if test_year in ["N1", "N2", "N3"] else "data"
        with archive.open(f"{folder}/{test_year}/caption.txt", "r") as f:
            caption_lines = [line.decode('utf-8').strip() for line in f.readlines()]
            for caption_line in caption_lines:
                caption_parts = caption_line.split()
                caption_file_name = caption_parts[0]
                caption_string = ' '.join(caption_parts[1:])
                caption[caption_file_name] = caption_string

    # Cálculo de la métrica ExpRate (tasa de expresiones correctas)
    with zipfile.ZipFile("result.zip") as archive:
        exprate=[0,0,0,0]  # Inicializa el contador de expresiones correctas por nivel de error
        file_list = archive.namelist()
        txt_files = [file for file in file_list if file.endswith('.txt')]
        for txt_file in txt_files:
            file_name = txt_file.rstrip('.txt')
            with archive.open(txt_file) as f:
                lines = f.readlines()
                pred_string = lines[1].decode('utf-8').strip()[1:-1]  # Predicción del modelo
                if file_name in caption:
                    caption_string = caption[file_name]  # Referencia de la expresión
                else:
                    print(file_name,"no encontrado en el archivo de captions")
                    continue
                caption_parts = caption_string.strip().split()
                pred_parts = pred_string.strip().split()
                if caption_string == pred_string:
                    exprate[0]+=1  # Incrementa el contador si la predicción es exacta
                else:
                    # Calcula la distancia de edición si la predicción no es exacta
                    error_num=cal_distance(pred_parts,caption_parts)
                    if error_num<=3:
                        exprate[error_num]+=1
        tot = len(txt_files)  # Total de archivos evaluados
        exprate_final=[]
        for i in range(1,5):
            # Calcula la tasa acumulada de expresiones correctas para cada nivel de error
            exprate_final.append(100*sum(exprate[:i])/tot)
        print(test_year,"ExpRate",exprate_final)  # Imprime los resultados finales

if __name__ == "__main__":
    typer.run(main)  # Ejecuta la función principal con los argumentos proporcionados
