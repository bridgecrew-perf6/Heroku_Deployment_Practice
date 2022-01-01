Final App link:
https://breast-cancer-demo-predictor.herokuapp.com/predict


"""Practice Project to deploy ML model on Heroku Platform with the help of FLASK framework"""
"""Business Problem/Goal of the ML project:
To Determine whether patient has malignant tumor or benign tumor was the semi-automated and lengthy process in practice. The older process leads to manual intervention in result of threats and challenges like bad accuracy, exra time taken, overwait, overprocessing, duplicasy of works, waiting time in the cycle. To overcome from these challenges and threats using machine learning technique to create Cancer Detection ML model to classify malignant and benign tumor from the given features."""

import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('Model_Cancer', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    input_features = [float(x) for x in request.form.values()]
    features_value = [np.array(input_features)]
    
    features_name = ['mean radius', 'mean texture', 'mean perimeter', 'mean area',
       'mean smoothness', 'mean compactness', 'mean concavity',
       'mean concave points', 'mean symmetry', 'mean fractal dimension',
       'radius error', 'texture error', 'perimeter error', 'area error',
       'smoothness error', 'compactness error', 'concavity error',
       'concave points error', 'symmetry error', 'fractal dimension error',
       'worst radius', 'worst texture', 'worst perimeter', 'worst area',
       'worst smoothness', 'worst compactness', 'worst concavity',
       'worst concave points', 'worst symmetry', 'worst fractal dimension']
    
    df = pd.DataFrame(features_value, columns=features_name)
    output = model.predict(df)
        
    if output == 0:
        res_val = "Breast Cancer"
    else:
        res_val = "No Breast Cancer"
        

    return render_template('index.html', prediction_text='Patient has {}'.format(res_val))

if __name__ == "__main__":
    app.run()

"""After Model Deployment need to Monitor the Model and maintain necessary retrainings to achive maximum accuracy. """
"""Conclusion: With the help of ML Model we can determine whether patient has malignant tumor or benign tumor and eliminate the older lengthy processes which was in practice. With the help of ML Model we can reduce the manual intervention and eliminate the threats and challenges like error in prediction, exra time taken, overwait, overprocessing, duplicasy of works, waiting time in the cycle."""
