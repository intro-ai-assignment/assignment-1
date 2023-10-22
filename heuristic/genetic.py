import random

def generate_chromosome(N):
    return [random.randint(0, N - 1) for value in range(N)]

def calc_fitness(chromosome, max_fitness):
    conflicts = 0
    N = len(chromosome)
    for i in range(N - 1):
        for j in range(i + 1, N):
            if chromosome[j] == chromosome[i] or abs(j - i) == abs(chromosome[j] - chromosome[i]):
                conflicts += 1
    return max_fitness - conflicts

def tournament_selection(population):
    tournament_size = 5
    tournament = random.sample(population, tournament_size)
    return max(tournament, key = lambda data: data[1])[0]

def crossover(parent1, parent2, N):
    cross_point = random.randint(1, N - 1)
    child1 = parent1[:cross_point] + parent2[cross_point:]
    child2 = parent1[cross_point:] + parent2[:cross_point]
    return (child1, child2)

def mutate(chromosome):
    N = len(chromosome)
    pos1, pos2 = random.sample(range(N), 2)
    chromosome[pos1], chromosome[pos2] = chromosome[pos2], chromosome[pos1]
    return chromosome

def genetic_alg(N):
    print("\t--- Genetic Algorithm ---")
    population_size = int(input("Enter population size = "))
    mutation_rate = 0.3
    max_generations = 100
    best_chromosome = []
    max_fitness = int(N * (N - 1) / 2)
    population = [generate_chromosome(N) for value in range(population_size)]
    for generation in range(max_generations):
        population = [(chromosome, calc_fitness(chromosome, max_fitness)) for chromosome in population]
        best_chromosome = max(population, key = lambda data: data[1])
        print(f"Generations: {generation + 1} - Fitness value: {best_chromosome[1]} - Max fitness: {max_fitness}")
        if best_chromosome[1] == max_fitness:
            print("Solution found in generation", generation + 1)
            break
        new_population = []
        new_population.append(best_chromosome[0])
        while len(new_population) < population_size:
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)
            childs = crossover(parent1, parent2, N)
            for child in childs:
                if random.random() < mutation_rate:
                    child = mutate(child)
                new_population.append(child)
        population = new_population
    print("Best solution:", best_chromosome)
    return best_chromosome[0]

    
    
    
    
