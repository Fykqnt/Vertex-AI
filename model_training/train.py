import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from prepare_data import load_data, prepare_data

def train_model(X_train, y_train):
    model = MLPRegressor(hidden_layer_sizes=(128, 64, 32),
                         activation='relu',
                         solver='adam',
                         batch_size=256,
                         learning_rate_init=0.001,
                         max_iter=100,
                         random_state=42,
                         verbose=True)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_val, y_val):
    y_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    print(f'Mean Squared Error: {mse}')

def save_model(model, path):
    with open(path, 'wb') as f:
        pickle.dump(model, f)

if __name__ == '__main__':
    interactions, user_vectors, item_vectors = load_data()
    X_train, y_train, X_val, y_val = prepare_data(interactions, user_vectors, item_vectors)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_val, y_val)
    save_model(model, 'path/to/save/model.pkl')

