import numpy as np
from genetic.gen import GeneticNeuralNetwork

if __name__ == "__main__":
    X = np.array([[0,0], [0,1], [1,0], [1,1]])
    Y = np.array([0, 1, 1, 0]).reshape(-1, 1)
    gnn = GeneticNeuralNetwork(layer_sizes=[2, 5, 5, 1])
    gnn.train(X, Y)
    print("Predictions:", gnn.predict(X))
