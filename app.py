import numpy as np
import pickle
from flask import Flask, render_template,request, jsonify, redirect, url_for

app = Flask(__name__)
classifier = pickle.load(open('classifier.pkl', 'rb'))
cv = pickle.load(open('cv.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        data = cv.transform(data).toarray()
        prediction = classifier.predict(data)
        return render_template('predict.html', prediction = prediction)
    

if __name__ == "__main__":
    app.run(debug=True)
