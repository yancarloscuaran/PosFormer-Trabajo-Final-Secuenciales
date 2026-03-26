from pytorch_lightning.plugins.training_type.ddp import DDPPlugin
from pytorch_lightning.utilities.cli import LightningCLI
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
from Pos_Former.datamodule import CROHMEDatamodule
from Pos_Former.lit_posformer import LitPosFormer
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging
from pathlib import Path

class MyLightningCLI(LightningCLI):
    """
    Clase personalizada para configurar y ejecutar el entrenamiento del modelo.
    Permite agregar argumentos adicionales y configurar callbacks y loggers.
    """
    def add_arguments_to_parser(self, parser):
        # Agrega un argumento para especificar el checkpoint del modelo
        parser.add_argument('--ckpt_path', type=str, default=None, help='Checkpoint path for the model')

    def before_fit(self):
        # Configura el directorio de trabajo y los callbacks antes de iniciar el entrenamiento
        if self.config['ckpt_path'] is None:
            cwd = self.trainer.default_root_dir
        else:
            cwd = str(Path(self.config['ckpt_path']).parents[1].absolute())

        # Callback para guardar los mejores checkpoints basados en la métrica de validación
        checkpoint = ModelCheckpoint(monitor='val_ExpRate', mode='max', save_top_k=1, save_last=True,
                                     filename='{epoch}-{step}-{val_ExpRate:.4f}')
        # Logger para visualizar métricas en TensorBoard
        logger = TensorBoardLogger(cwd, '', '.')
        self.trainer.callbacks.extend([checkpoint])
        self.trainer.logger = logger
        self.trainer.enable_model_summary = True

# Punto de entrada principal para configurar y ejecutar el entrenamiento
cli = MyLightningCLI(
    LitPosFormer,  # Modelo a entrenar
    CROHMEDatamodule,  # Módulo de datos para cargar y procesar el dataset
    save_config_overwrite=True,  # Permite sobrescribir la configuración guardada
    trainer_defaults={"plugins": DDPPlugin(find_unused_parameters=True)},  # Configuración del entrenador
)

# Nota: En este proyecto, este archivo no se ejecutó porque se utilizaron pesos preentrenados
# provistos por los autores del artículo. Sin embargo, este script permite entrenar el modelo
# desde cero configurando hiperparámetros como el learning rate y el batch size, que son
# críticos para el rendimiento del modelo.