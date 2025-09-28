import pickle
import numpy as np

# Load the model once when this file is imported
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

def predict_crop(N, P, K, temperature, humidity, pH, rainfall):
    """
    Inputs: all numeric features from the form
    Output: predicted crop (string)
    """
    data = np.array([[N, P, K, temperature, humidity, pH, rainfall]])
    prediction = model.predict(data)
    return prediction[0]

