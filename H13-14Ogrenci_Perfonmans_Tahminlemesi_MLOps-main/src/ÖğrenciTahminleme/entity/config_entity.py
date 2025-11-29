from pathlib import Path
from dataclasses import dataclass

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path
    raw_data_path: Path
    train_data_path: Path
    test_data_path: Path


@dataclass(frozen=True)
class DataTransformConfig:
    root_dir: Path  # Türü belirtmek için ':' kullanın
    preprocessor_file: Path  # Türü belirtmek için ':' kullanın
    data_path: Path  # Türü belirtmek için ':' kullanın

@dataclass
class ModelTrainerConfig:
    root_dir: Path
    prep_data_path: Path
    model_file: Path