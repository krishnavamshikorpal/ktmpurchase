# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 16:53:40 2020

@author: Krishna Vamshi
"""

import numpy as np
import pickle
from flask import Flask, render_template,jsonify,request

##Intializing the flask application
app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

##Routing the application to root folder
@app.route('/')
def home():
    return render_template('ktm.html')

##Routing the prediction outcome
@app.route('/predict',methods=['POST'])
def predict():
    int_features = [int(x1) for x1 in request.form.values()]
    final_data = [np.array(int_features)]
    prediction = model.predict_proba(final_data)
    output = prediction[0][1]
    return render_template('ktm.html',prediction_text='chances for purchasing the bike {}'.format(round(output,2)))

if __name__=='__main__':
    app.run(debug=True)
    
