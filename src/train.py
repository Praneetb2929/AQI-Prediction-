# src/train.py

import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from preprocess import load_data, clean_data, get_features

def train_model():
    df = load_data("data/city_day.csv")
    df = clean_data(df)

    X, y = get_features(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = LinearRegression()
    model.fit(X_train, y_train)

    with open("models/model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("Model trained and saved!")

if __name__ == "__main__":
    train_model()