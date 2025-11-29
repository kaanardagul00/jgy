from src.ÖğrenciTahminleme.config.configration import ConfigurationManager
from src.ÖğrenciTahminleme.components.model_trainer import ModelTrainer
from src.ÖğrenciTahminleme.components.data_transformation import DataTransfrom

from src.ÖğrenciTahminleme import logger
from src.ÖğrenciTahminleme.pipeline.stage_02_data_transformation import DataTransformationTrainingPipeline
STAGE_NAME = "Model Trainer stage"  # Aşama ismi

class ModelTrainerTrainingPipeline:
    def __init__(self):
        pass

    def main(self, train_arr, test_arr):
        # Konfigürasyon yöneticisini oluştur
        config = ConfigurationManager()
        # Model eğitici konfigürasyonunu al
        model_trainer_config = config.get_model_trainer_config()
        # ModelTrainer sınıfını başlat
        model_trainer = ModelTrainer(config=model_trainer_config)
        # Model eğitimi başlat
        model_trainer.initiate_model_trainer(train_arr, test_arr)