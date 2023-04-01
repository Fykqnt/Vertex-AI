import numpy as np
import pandas as pd
import pickle
from prepare_data import load_data, prepare_data

def load_model(path):
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model

