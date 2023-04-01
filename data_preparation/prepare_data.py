import pandas as pd
import numpy as np

def load_data():
    # Load interaction data, user vectors, and item vectors
    # You might need to adjust this part based on your data sources
    interactions = pd.read_csv('path/to/interactions.csv')
    user_vectors = pd.read_csv('path/to/user_vectors.csv')
    item_vectors = pd.read_csv('path/to/item_vectors.csv')

    return interactions, user_vectors, item_vectors

def prepare_data(interactions, user_vectors, item_vectors):
    # Merge interaction data with user and item vectors
    data = interactions.merge(user_vectors, on='user_id').merge(item_vectors, on='item_id')

    # Split data into training and validation sets
    train_data = data.sample(frac=0.8, random_state=42)
    val_data = data.drop(train_data.index)

    # Separate features and target variable
    X_train = train_data.drop('rating', axis=1)
    y_train = train_data['rating']

    X_val = val_data.drop('rating', axis=1)
    y_val = val_data['rating']

    return X_train, y_train, X_val, y_val

