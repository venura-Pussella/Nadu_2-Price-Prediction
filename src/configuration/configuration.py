# src/configuration.py
from pathlib import Path
from src.constants import *
from src.utils.common import read_yaml
from src.entity.config_entity import DataIngestionConfig, DataValidationConfig, DataPreprocessingConfig, ModelEvaluationConfig, ModelTrainerConfig


def load_configuration(config_filepath: Path = CONFIG_FILE_PATH, schema_filepath: Path = SCHEMA_FILE_PATH):

    config = read_yaml(config_filepath)
    schema = read_yaml(schema_filepath)
    return config, schema

###>>> Data Ingestion Configuration Start <<<###
def get_data_ingestion_config(config) -> DataIngestionConfig:

    # Extract data ingestion settings from the config
    data_ingestion = config.data_ingestion

    # Create and return a DataIngestionConfig instance
    return DataIngestionConfig(
        root_dir=data_ingestion.root_dir,
        source_URL=data_ingestion.source_URL,
        local_data_file=data_ingestion.local_data_file,
        unzip_dir=data_ingestion.unzip_dir
    )
###>>> Data Ingestion Configuration End <<<###

##>>> Data Valiadation Configuration Start <<<###
def get_data_validation_config(config, schema) -> DataValidationConfig:
    
    data_validation = config.data_validation

    # Create and return a DataValidationConfig instance
    return DataValidationConfig(
        root_dir=data_validation.root_dir,
        unzip_data_dir=data_validation.unzip_data_dir,
        STATUS_FILE=data_validation.STATUS_FILE,
        all_schema=schema.COLUMNS,
    )
###>>> Data Validation Configuration End <<<###

## Data Preprocessing Configuration Start ##
def get_data_preprocessing_config(config) -> DataPreprocessingConfig:

    # Extract data validation settings from the config
    data_preprocessing = config.data_preprocessing

    # Create and return a DataValidationConfig instance, including the schema
    return DataPreprocessingConfig(
        root_dir=data_preprocessing.root_dir,
        unzip_data_dir=data_preprocessing.unzip_data_dir,
        local_data_file=data_preprocessing.local_data_file,
    )
## Data Preprocessing Configuration End ##




## Data Transformation Configuration ##

## Model Training Configuration ##

## Model Evaluation Configuration ##
