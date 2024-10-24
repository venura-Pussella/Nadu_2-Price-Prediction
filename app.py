from flask import Flask, render_template, request
import os
from prediction import prediction_pipeline  # Adjust the import to match your project structure

app = Flask(__name__)  # initializing a Flask app

@app.route('/', methods=['GET'])  # route to display the home page
def home_page():
    return render_template("index.html")

@app.route('/train', methods=['GET'])  # route to train the pipeline
def training():
    try:
        # Call your training script (make sure it is callable)
        os.system("python main.py")
        return "Training Successful!"
    except Exception as e:
        return f"Error during training: {str(e)}"

@app.route('/predict', methods=['POST', 'GET'])  # route to show the predictions in a web UI
def index():
    if request.method == 'POST':
        try:
            # Reading form data from the POST request
            prices_str = request.form['prices']
            
            # Split the input by comma, space, or new lines and convert it into a list of floats
            input_prices = [float(price) for price in prices_str.replace('\n', ',').replace(' ', ',').split(',') if price.strip()]
            
            if len(input_prices) != 30:
                return "Please enter exactly 30 prices."
            
            # Making predictions using the PredictionPipeline
            predicted_price = prediction_pipeline(input_prices)
            
            # Rendering the template with the prediction result
            return render_template('results.html', prediction=str(predicted_price))

        except Exception as e:
            return f"Error during prediction: {str(e)}"
    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
