#2#the perceptron model can be applied for any dataset. Perceptron model is the basic model . So the prediction is wrong .Insted of 1101 it gave 1111 coz perceptron is the basic model.
from Perceptron import  Perceptron
import numpy as np


if __name__ == "__main__":
    
    # OR gate dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 1])  

    perceptron = Perceptron(epochs=10)
    perceptron.fit(X, y)

    test_data = np.array([[0, 1], [1, 0], [0, 0], [1, 0]])#unseen data
    predictions = perceptron.predict(test_data)

    print("Predictions :", predictions)