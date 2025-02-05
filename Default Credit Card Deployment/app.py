#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask, request, render_template
import numpy as np
import pickle

# Load the trained model and scaler
with open('rf_model.pkl', 'rb') as model_file:
    rf_model = pickle.load(model_file)
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    features = np.array(features).reshape(1, -1)
    scaled_features = scaler.transform(features)
    prediction = rf_model.predict(scaled_features)
    return render_template('index.html', prediction_text='Predicted Class: {}'.format(prediction[0]))

if __name__ == "__main__":
    app.run(debug=True)


# In[ ]:




