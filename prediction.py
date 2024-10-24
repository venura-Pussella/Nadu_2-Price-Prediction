from src import logger
import tensorflow as tf
from src.pipeline.inference.inference_stage_01_preprocessing import box_cox_with_min_max_scaling
from src.pipeline.inference.inference_stage_02_output_preprocessing import inverse_transform_output

def prediction_pipeline():
    
    try:
        # # Box_cox transform
        input_sequence = box_cox_with_min_max_scaling()

        # Load your trained LSTM model
        model = tf.keras.models.load_model('artifacts/model_trainer/model/best_model.keras')

        # make prediction
        prediction = model.predict(input_sequence)

        # Inverse transform the prediction to see real values
        true_price = inverse_transform_output(prediction)

        print(true_price)

    except Exception as e:
        logger.error(f"An error occurred during inference pipeline: {e}")

if __name__ == "__main__":

 prediction_pipeline()