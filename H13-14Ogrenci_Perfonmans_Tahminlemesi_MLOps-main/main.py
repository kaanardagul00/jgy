from src.ÖğrenciTahminleme import logger
from src.ÖğrenciTahminleme.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from src.ÖğrenciTahminleme.pipeline.stage_02_data_transformation import DataTransformationTrainingPipeline
from src.ÖğrenciTahminleme.pipeline.stage_03_model_trainer import ModelTrainerTrainingPipeline

STAGE_NAME1 = 'data_ingestion'
STAGE_NAME2 = 'data_transformation'
STAGE_NAME3 = 'model_trainer'

if __name__ == '__main__':
    try:
        # Aşamanın başladığını kaydet
        logger.info(f">>>>>> stage {STAGE_NAME1} started <<<<<<")
        data_ingestion_pipeline = DataIngestionTrainingPipeline()  # Pipeline nesnesini oluştur
        train_data, test_data = data_ingestion_pipeline.main()  # Ana fonksiyonu çalıştır
        # Aşamanın tamamlandığını kaydet
        logger.info(f">>>>>> stage {STAGE_NAME1} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)  # Hata durumunda kaydet
        raise e  # Hata fırlat
    
    try:
        # Aşamanın başladığını kaydet
        logger.info(f">>>>>> stage {STAGE_NAME2} started <<<<<<")
        data_transformation_pipeline = DataTransformationTrainingPipeline()  # Pipeline nesnesini oluştur
        train_arr, test_arr = data_transformation_pipeline.main(train_data, test_data)  # Ana fonksiyonu çalıştır
        # Aşamanın tamamlandığını kaydet
        logger.info(f">>>>>> stage {STAGE_NAME2} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)  # Hata durumunda kaydet
        raise e  # Hata fırlat
    
    try:
        # Aşamanın başladığını kaydet
        logger.info(f">>>>>> stage {STAGE_NAME3} started <<<<<<")
        model_trainer_pipeline = ModelTrainerTrainingPipeline()  # Pipeline nesnesini oluştur
        model_trainer_pipeline.main(train_arr, test_arr)  # Ana fonksiyonu çalıştır
        # Aşamanın tamamlandığını kaydet
        logger.info(f">>>>>> stage {STAGE_NAME3} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)  # Hata durumunda kaydet
        raise e  # Hata fırlat