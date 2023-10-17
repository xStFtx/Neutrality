import numpy as np
from genetic.gen import GeneticNeuralNetwork

if __name__ == "__main__":
    X = np.array([[0,0], [0,1], [1,0], [1,1]])
    Y = np.array([0, 1, 1, 0])  # XOR problem
    gnn = GeneticNeuralNetwork()
    gnn.train(X, Y)
    print("Best Weights:", gnn.best_weights)
    print("Predictions:", gnn.predict(X))
