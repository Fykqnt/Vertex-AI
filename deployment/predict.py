import numpy as np
import pandas as pd
import pickle
from prepare_data import load_data, prepare_data

def load_model(path):
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model

def generate_rankings(model, user_vector, item_vectors):
    # Create a DataFrame with user_vector replicated for each item
    user_vector_df = pd.DataFrame(np.tile(user_vector, (len(item_vectors), 1)))

    # Concatenate user_vector_df and item_vectors
    input_data = pd.concat([user_vector_df, item_vectors], axis=1)

    # Predict rankings
    rankings = model.predict(input_data)

    # Add rankings to item_vectors and sort by rankings
    item_vectors['ranking'] = rankings
    sorted_items = item_vectors.sort_values(by='ranking', ascending=False)

    return sorted_items

if __name__ == '__main__':
    model_path = 'path/to/saved/model.pkl'
    model = load_model(model_path)

    # Load user_vectors and item_vectors
    _, user_vectors, item_vectors = load_data()

    # Example: Generate personalized rankings for a specific user
    user_id = 1
    user_vector = user_vectors[user_vectors['user_id'] == user_id].drop('user_id', axis=1).values[0]
    personalized_rankings = generate_rankings(model, user_vector, item_vectors)
    print(personalized_rankings)

