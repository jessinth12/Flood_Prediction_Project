from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler using joblib
model = joblib.load("floods.save")
scaler = joblib.load("transform.save")

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict_page')
def predict_page():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    final = scaler.transform([features])
    prediction = model.predict(final)

    if prediction[0] == 1:
        return render_template("chance.html")
    else:
        return render_template("nochance.html")

if __name__ == "__main__":
    app.run(debug=True)