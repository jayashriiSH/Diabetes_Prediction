from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('diabetes_model.sav', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the form
    features = [float(x) for x in request.form.values()]
    input_data = np.array([features])

    # Make prediction
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        result = ' The person is likely Diabetic.'
    else:
        result = ' The person is likely Non-Diabetic.'

    return render_template('index.html', prediction_text=result)

if __name__ == '__main__':
    app.run(debug=True)
