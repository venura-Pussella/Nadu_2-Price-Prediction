from src import logger
from src.components.model_trainer import sequence_creation_train_test_split, save_train_test_data_to_excel, lstm_model_trainer
from src.utils.common import create_directories
from src.configuration.configuration import load_configuration, get_model_trainer_config

from src import logger

def model_trainer_training_pipeline():
    try:
        # Load config and schema
        config,schema = load_configuration()

        # Retrieve the data ingestion configuration from the loaded config
        model_trainer_config = get_model_trainer_config(config,schema)

        # Create directories related to data ingestion (root directory)
        create_directories([model_trainer_config.root_dir, model_trainer_config.root_dir_train, model_trainer_config.root_dir_test])

        # Sequence creation
        train_x, test_x, train_y, test_y = sequence_creation_train_test_split(model_trainer_config)
        
        # Saving data to an excel from numpy array (3D to 2D)
        save_train_test_data_to_excel(train_x, test_x, train_y, test_y, model_trainer_config)

        # Train the model
        lstm_model_trainer(train_x, train_y, model_trainer_config)

    except Exception as e:
        logger.error(f"An error occurred in model trainer : {e}")

if __name__ == "__main__":

    model_trainer_training_pipeline()