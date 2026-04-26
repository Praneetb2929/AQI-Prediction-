# src/evaluate.py

import pickle
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from preprocess import load_data, clean_data, get_features

def evaluate_model():
    df = load_data("data/city_day.csv")
    df = clean_data(df)

    X, y = get_features(df)

    model = pickle.load(open("models/model.pkl", "rb"))

    y_pred = model.predict(X)

    print("R2:", r2_score(y, y_pred))
    print("MAE:", mean_absolute_error(y, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y, y_pred)))

if __name__ == "__main__":
    evaluate_model()