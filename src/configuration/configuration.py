# src/configuration.py
from pathlib import Path
from src.constants import *
from src.utils.common import read_yaml
from src.entity.config_entity import DataIngestionConfig, DataValidationConfig, DataPreprocessingConfig, DataTransformationConfig, ModelTrainerConfig


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

###>>> Data Preprocessing Configuration Start <<<###
def get_data_preprocessing_config(config) -> DataPreprocessingConfig:

    # Extract data validation settings from the config
    data_preprocessing = config.data_preprocessing

    # Create and return a DataValidationConfig instance, including the schema
    return DataPreprocessingConfig(
        root_dir=data_preprocessing.root_dir,
        unzip_data_dir=data_preprocessing.unzip_data_dir,
        local_data_file=data_preprocessing.local_data_file,
    )
###>>> Data Preprocessing Configuration End <<<###


###>>> Data Transformation Configuration Start <<<###

# Function to get data ingestion configuration from the loaded config
def get_data_transformation_config(config) -> DataTransformationConfig:

    # Extract data ingestion settings from the config
    data_transformation = config.data_transformation

    # Create and return a DataIngestionConfig instance
    return DataTransformationConfig(
        root_dir=data_transformation.root_dir,
        data_path=data_transformation.data_path,
        local_data_file=data_transformation.local_data_file
    )
###>>> Data Transformation Configuration End <<<###


##>>> Model Training Configuration Start <<<##
# Function to initialize configuration by reading YAML and creating directories
def load_configuration(config_filepath: Path = CONFIG_FILE_PATH, schema_filepath: Path = SCHEMA_FILE_PATH):

    config = read_yaml(config_filepath)
    schema = read_yaml(schema_filepath)

    return config, schema

# Function to get data ingestion configuration from the loaded config
def get_model_trainer_config(config,schema) -> ModelTrainerConfig:

    # Extract data ingestion settings from the config
    model_trainer = config.model_trainer

    # Create and return a DataIngestionConfig instance
    return ModelTrainerConfig(
        root_dir=model_trainer.root_dir,
        local_data_path=model_trainer.local_data_path,
        train_x_data_file=model_trainer.train_x_data_file,
        train_y_data_file=model_trainer.train_y_data_file,
        test_x_data_file=model_trainer.test_x_data_file,
        test_y_data_file=model_trainer.test_y_data_file,
        model_checkpoint_path=model_trainer.model_checkpoint_path,
        model_name=model_trainer.model_name,
        n_units_layer1=model_trainer.n_units_layer1,
        n_units_layer2=model_trainer.n_units_layer2,
        n_units_layer3=model_trainer.n_units_layer3,
        dropout_rate=model_trainer.dropout_rate,
        sequence_length=model_trainer.sequence_length,
        optimizer=model_trainer.optimizer,
        loss_function=model_trainer.loss_function,
        epochs=model_trainer.epochs,
        batch_size=model_trainer.batch_size,
        validation_split=model_trainer.validation_split,
        target_column = schema.TARGET_COLUMN
    )
##>>> Model Training Configuration End <<<##




## Model Evaluation Configuration ##
