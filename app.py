
from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)


model = pickle.load(open('loan_model.pkl', 'rb'))
with open('features.pkl', 'rb') as features_file:
    feature_names = pickle.load(features_file)


LABELS = {0: 'Not Approved', 1: 'Approved'}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        features = np.array([[int(request.form['gender']),
                              int(request.form['married']),
                              int(request.form['dependents']),
                              int(request.form['education']),
                              int(request.form['self_employed']),
                              int(request.form['applicant_income']),
                              float(request.form['coapplicant_income']),
                              float(request.form['loan_amount']),
                              float(request.form['loan_amount_term']),
                              float(request.form['credit_history']),
                              int(request.form['property_area'])]])

        
        features_df = pd.DataFrame(features, columns=feature_names)

       
        prediction = model.predict(features_df)
        result = LABELS.get(prediction[0], 'Unknown')

        return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
