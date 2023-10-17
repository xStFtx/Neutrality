import numpy as np

# Define the neural network structure
def neural_net(inputs, weights):
    return np.dot(inputs, weights)

# Fitness function
def fitness(individual, X, Y):
    predictions = neural_net(X, individual)
    return -np.mean(np.square(predictions - Y))

# Crossover function
# Crossover function
def crossover(parent1, parent2):
    if len(parent1) <= 2:
        return parent1, parent2  # Just return parents if length is 1 or 2

    point = np.random.randint(1, len(parent1)-1)
    child1 = np.concatenate([parent1[:point], parent2[point:]])
    child2 = np.concatenate([parent2[:point], parent1[point:]])
    return child1, child2


# Mutation function
def mutate(individual, mutation_rate=0.01):
    for i in range(len(individual)):
        if np.random.rand() < mutation_rate:
            individual[i] += np.random.normal()
    return individual

# Genetic algorithm
def genetic_neural_network(X, Y, population_size=100, generations=1000):
    # Initialize population
    population = [np.random.randn(X.shape[1]) for _ in range(population_size)]
    
    for generation in range(generations):
        # Evaluate fitness
        fitnesses = [fitness(ind, X, Y) for ind in population]
        
        # Select parents based on fitness
        parents = np.argsort(fitnesses)[-2:]
        
        # Create offspring through crossover
        child1, child2 = crossover(population[parents[0]], population[parents[1]])
        
        # Mutate offspring
        child1 = mutate(child1)
        child2 = mutate(child2)
        
        # Replace two worst individuals with children
        worst = np.argsort(fitnesses)[:2]
        population[worst[0]] = child1
        population[worst[1]] = child2
        
    # Return the best individual
    best_index = np.argmax(fitnesses)
    return population[best_index]

# Example usage
X = np.array([[0,0], [0,1], [1,0], [1,1]])
Y = np.array([0, 1, 1, 0])  # XOR problem
best_weights = genetic_neural_network(X, Y)

print("Best Weights:", best_weights)
print("Predictions:", neural_net(X, best_weights))
