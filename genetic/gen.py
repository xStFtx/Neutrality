import numpy as np

class GeneticNeuralNetwork:
    def __init__(self, population_size=100, generations=1000):
        self.population_size = population_size
        self.generations = generations

    def _neural_net(self, inputs, weights):
        return np.dot(inputs, weights)

    def _fitness(self, individual, X, Y):
        predictions = self._neural_net(X, individual)
        return -np.mean(np.square(predictions - Y))

    def _crossover(self, parent1, parent2):
        if len(parent1) <= 2:
            return parent1, parent2  # Just return parents if length is 1 or 2

        point = np.random.randint(1, len(parent1)-1)
        child1 = np.concatenate([parent1[:point], parent2[point:]])
        child2 = np.concatenate([parent2[:point], parent1[point:]])
        return child1, child2

    def _mutate(self, individual, mutation_rate=0.01):
        for i in range(len(individual)):
            if np.random.rand() < mutation_rate:
                individual[i] += np.random.normal()
        return individual

    def train(self, X, Y):
        # Initialize population
        population = [np.random.randn(X.shape[1]) for _ in range(self.population_size)]
        
        for generation in range(self.generations):
            # Evaluate fitness
            fitnesses = [self._fitness(ind, X, Y) for ind in population]
            
            # Select parents based on fitness
            parents = np.argsort(fitnesses)[-2:]
            
            # Create offspring through crossover
            child1, child2 = self._crossover(population[parents[0]], population[parents[1]])
            
            # Mutate offspring
            child1 = self._mutate(child1)
            child2 = self._mutate(child2)
            
            # Replace two worst individuals with children
            worst = np.argsort(fitnesses)[:2]
            population[worst[0]] = child1
            population[worst[1]] = child2
        
        # Return the best individual
        best_index = np.argmax(fitnesses)
        self.best_weights = population[best_index]

    def predict(self, X):
        return self._neural_net(X, self.best_weights)


# Usage
if __name__ == "__main__":
    X = np.array([[0,0], [0,1], [1,0], [1,1]])
    Y = np.array([0, 1, 1, 0])  # XOR problem
    gnn = GeneticNeuralNetwork()
    gnn.train(X, Y)
    print("Best Weights:", gnn.best_weights)
    print("Predictions:", gnn.predict(X))
