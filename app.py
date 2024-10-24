from flask import Flask, render_template, request
import os 
import numpy as np
import pandas as pd
from prediction import PredictionPipeline


app = Flask(__name__) # initializing a flask app


@app.route('/',methods=['GET'])  # route to display the home page
def homePage():
    return render_template("index.html")

@app.route('/train',methods=['GET'])  # route to train the pipeline
def training():
    os.system("python main.py")
    return "Training Successful!" 

@app.route('/predict', methods=['POST', 'GET'])  # Route to show predictions in a web UI
def index():
    if request.method == 'POST':
        try:
            # Reading 30 days of food prices from the form data
            prices = []
            for day in range(1, 31):
                price = float(request.form[f'Day{day}'])  # Extract price for each day
                prices.append(price)
            
            # Step 1: Box-Cox transformation
            transformed_prices = box_cox_transform_inference(prices, lambda_value)
            
            # Step 2: Min-Max scaling
            scaled_prices = min_max_scale_inference(transformed_prices, scaler)
            
            # Reshape the data to match the input shape of your model (e.g., [1, 30])
            data = np.array(scaled_prices).reshape(1, -1)
            
            # Step 3: Making predictions using the PredictionPipeline (assumed to be implemented)
            obj = PredictionPipeline()  # Assume the PredictionPipeline is your model wrapper
            predict = obj.predict(data)
            
            # Rendering the template with the prediction result
            return render_template('results.html', prediction=str(predict))

        except Exception as e:
            print('The Exception message is: ', e)
            return 'Something went wrong during prediction.'

    else:
        # Render the 'index.html' template for GET requests
        return render_template('index.html')


if __name__ == "__main__":
	    app.run(host="0.0.0.0", port = 8080)