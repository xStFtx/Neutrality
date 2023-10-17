import numpy as np
import multiprocessing

def relu(x):  
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

ACTIVATIONS = [np.tanh, sigmoid, relu]
ACTIVATION_NAMES = ["tanh", "sigmoid", "relu"]

class GeneticNeuralNetwork:
    def __init__(self, layer_sizes, population_size=100, generations=1000, elitism_ratio=0.1, mutation_range=0.5, tournament_size=5, dropout_rate=0.1, l2_lambda=0.001):
        self.layer_sizes = layer_sizes
        self.population_size = population_size
        self.generations = generations
        self.elitism_ratio = int(elitism_ratio * population_size)
        self.mutation_range = mutation_range
        self.tournament_size = tournament_size
        self.dropout_rate = dropout_rate
        self.l2_lambda = l2_lambda
        self.starting_mutation_range = mutation_range

    def _dropout(self, matrix):
        dropout_mask = np.random.rand(*matrix.shape) > self.dropout_rate
        return matrix * dropout_mask

    def _neural_net(self, inputs, genome):
        idx = 0
        current_input = inputs
        
        for l in range(len(self.layer_sizes) - 1):
            W_size = self.layer_sizes[l] * self.layer_sizes[l+1]
            W = genome[idx:idx+W_size].reshape(self.layer_sizes[l], self.layer_sizes[l+1])
            idx += W_size
            
            b = genome[idx:idx+self.layer_sizes[l+1]]
            idx += self.layer_sizes[l+1]

            activation_idx = int(genome[idx])
            idx += 1
            
            z = np.dot(current_input, W) + b
            current_input = ACTIVATIONS[activation_idx](z)
            current_input = self._dropout(current_input)
        
        return current_input

    def _fitness(self, genome, X, Y):
        predictions = self._neural_net(X, genome)
        mse = -np.mean(np.square(predictions - Y))
        l2_regularization = -self.l2_lambda * np.sum(np.square(genome))
        return mse + l2_regularization

    def _crossover(self, parent1, parent2):
        crossover_points = sorted([0] + [np.random.randint(1, len(parent1)-1) for _ in range(np.random.randint(1, 4))])
        offspring = np.zeros_like(parent1)
        flip = True
        for i in range(len(crossover_points) - 1):
            if flip:
                offspring[crossover_points[i]:crossover_points[i+1]] = parent1[crossover_points[i]:crossover_points[i+1]]
            else:
                offspring[crossover_points[i]:crossover_points[i+1]] = parent2[crossover_points[i]:crossover_points[i+1]]
            flip = not flip
        return offspring

    def _mutate(self, genome):
        for i in range(len(genome)):
            if np.random.rand() < (self.mutation_range / len(genome)):
                if i % (sum(self.layer_sizes) + len(self.layer_sizes) - 1) == 0:  # Activation function index
                    genome[i] = np.random.choice(len(ACTIVATIONS)-1)
                else:
                    genome[i] += np.random.normal()
        return genome

    def _tournament_selection(self, population, fitnesses):
        candidates = np.random.choice(self.population_size, self.tournament_size)
        selected = max(candidates, key=lambda x: fitnesses[x])
        return population[selected]

    def _adaptive_mutation(self, stagnation_counter):
        if stagnation_counter > 25:
            self.mutation_range *= 1.1
        else:
            self.mutation_range = self.starting_mutation_range

    def _elitism_boost(self, population, fitnesses, boost_factor=2):
        sorted_indices = np.argsort(fitnesses)
        elites = [population[i] for i in sorted_indices[-self.elitism_ratio:]]
        return elites * boost_factor

    def _parallel_fitness(self, genome):
        return self._fitness(genome, self._X, self._Y)

    def train(self, X, Y, early_stopping_rounds=50):
        self._X = X  # For parallel processing
        self._Y = Y  # For parallel processing

        genome_length = sum(self.layer_sizes[i] * self.layer_sizes[i+1] + self.layer_sizes[i+1] for i in range(len(self.layer_sizes)-1)) + (len(self.layer_sizes)-1)
        population = [np.concatenate([np.random.randn(genome_length - (len(self.layer_sizes)-1)), np.random.choice(len(ACTIVATIONS)-1, len(self.layer_sizes)-1)]) for _ in range(self.population_size)]
        
        best_fitness = float("-inf")
        stagnation_counter = 0

        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())  # Parallel processing

        for generation in range(self.generations):
            fitnesses = pool.map(self._parallel_fitness, population)  # Parallel fitness computation

            if max(fitnesses) > best_fitness:
                best_fitness = max(fitnesses)
                stagnation_counter = 0
            else:
                stagnation_counter += 1
            self._adaptive_mutation(stagnation_counter)  # Adjust mutation rate based on progress

            if stagnation_counter > early_stopping_rounds:
                print(f"Training stopped early at generation {generation + 1}.")
                break

            new_population = self._elitism_boost(population, fitnesses)  # Enhanced elitism
            
            while len(new_population) < self.population_size:
                parent1 = self._tournament_selection(population, fitnesses)
                parent2 = self._tournament_selection(population, fitnesses)
                child = self._crossover(parent1, parent2)
                child = self._mutate(child)
                new_population.append(child)
            
            population = new_population[:self.population_size]
            if generation % 100 == 0 or generation == self.generations-1:
                print(f"Generation {generation + 1}, Best Fitness: {best_fitness:.6f}, Mutation Rate: {self.mutation_range:.4f}")

        self.best_genome = population[np.argmax(fitnesses)]
        pool.close()
        pool.join()

    def predict(self, X):
        return self._neural_net(X, self.best_genome)
