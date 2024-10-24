from src import logger
import tensorflow as tf
from src.pipeline.inference.inference_stage_01_preprocessing import box_cox_with_min_max_scaling
from src.pipeline.inference.inference_stage_02_output_preprocessing import inverse_transform_output

def prediction_pipeline(input_prices, lambda_value = -0.1207758043220706):
    
    try:
        # # Box_cox transform and Min_Max_Scaling
        input_sequence = box_cox_with_min_max_scaling(input_prices, lambda_value)

        # Load your trained LSTM model
        model = tf.keras.models.load_model('artifacts/model_trainer/model/best_model.keras')

        # make prediction
        prediction = model.predict(input_sequence)

        # Inverse transform the prediction to see real values
        true_price = inverse_transform_output(prediction, lambda_value)

        return true_price
    
    except Exception as e:
        logger.error(f"An error occurred during inference pipeline: {e}")

# if __name__ == "__main__":
#     # Example input: 30 days of pettah average prices
#     input_prices = [
#         100.0, 102.5, 105.0, 103.5, 101.0, 99.5, 98.0, 97.0, 96.0, 95.5,
#         96.5, 97.5, 98.5, 99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0,
#         106.0, 107.0, 108.0, 109.0, 110.0, 111.0, 112.0, 113.0, 114.0, 115.0
#     ]

#     # Run the prediction pipeline
#     predicted_price = prediction_pipeline(input_prices)
#     print(f"Predicted Price: {predicted_price}")