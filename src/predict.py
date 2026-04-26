# src/predict.py

import pickle
import numpy as np

model = pickle.load(open("models/model.pkl", "rb"))

def predict_aqi(pm25, pm10, no, no2, nox, nh3):
    input_data = np.array([[pm25, pm10, no, no2, nox, nh3]])
    prediction = model.predict(input_data)
    return prediction[0]