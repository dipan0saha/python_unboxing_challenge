from random import randint, SystemRandom, uniform
import numpy as np
from numpy.random import choice
import sys

MAX_ITER = 100000


class ChessBoard:
    def __init__(self):
        self.sequence = None
        self.fitness = None
        self.probability = None

    def set_sequence(self, sequence_value):
        self.sequence = sequence_value

    def set_fitness(self, fitness_value):
        self.fitness = fitness_value

    def set_probability(self, probability_value):
        self.probability = probability_value

    def get_sequence(self):
        return self.sequence

    def get_fitness(self):
        return self.fitness

    def get_probability(self):
        return self.probability


class SolveNQueens:
    def __init__(self):
        self.population_size = 100
        self.number_of_queens = None
        self.population = None
        self.mutation = 0.000001
        self.crossover = 0.5
        self.stop_fitness = 0
        self.secure_random = SystemRandom()
        self.alpha_list = None

    def set_number_of_queens(self, number_of_queens):
        self.number_of_queens = number_of_queens
        self.set_stop_fitness()
        self.alpha_list = [x for x in range(self.number_of_queens)]

    def set_population_size(self, population_size):
        self.population_size = population_size

    def set_mutation(self, mutation_value):
        self.mutation = mutation_value

    def set_stop_fitness(self):
        """ sets the maximum fitness value at which the iteration wil stop """
        for max_possible_conflicts in range(0, self.number_of_queens):
            self.stop_fitness += max_possible_conflicts

    def generate_chromosome(self):
        """ randomly generates a sequence of board states """
        init_distribution = np.arange(self.number_of_queens)
        np.random.shuffle(init_distribution)

        return init_distribution

    def calculate_individual_fitness(self, chromosome):
        """ returns [self.stop_fitness - <number of conflicts>] """
        total_clashes = 0

        # Calculate row & column clashes
        row_col_clashes = abs(len(chromosome) - len(np.unique(chromosome)))
        total_clashes += row_col_clashes

        # Calculate diagonal clashes
        for i in range(len(chromosome)):
            for j in range(len(chromosome)):
                if i != j:
                    dx = abs(i - j)
                    dy = abs(chromosome[i] - chromosome[j])
                    if dx == dy:
                        total_clashes += 1

        return self.stop_fitness - total_clashes

    def generate_population(self):
        """ generates the initial population"""
        population = [ChessBoard() for i in range(self.population_size)]

        for i in range(self.population_size):
            chromosome = self.generate_chromosome()
            population[i].set_sequence(chromosome)

        self.population = population

        self.calculate_fitness()
        self.calculate_probability()

    def calculate_fitness(self):
        """ calculates the fitness for the entire population"""
        for each in self.population:
            each.set_fitness(self.calculate_individual_fitness(each.get_sequence()))

    def calculate_probability(self):
        """ calculates the probability/score for the entire population"""
        for each in self.population:
            each.set_probability(each.fitness / (np.sum([x.fitness for x in self.population]) * 1.0))

    def get_parent(self):
        """ Get 2 parents """
        # Note! A parent is decided by random probability of survival
        while True:
            random_prob = np.random.rand()
            parent01_pool = [x for x in self.population if x.get_probability() <= random_prob]
            if parent01_pool:
                parent01 = parent01_pool[np.random.randint(len(parent01_pool))]
                break
        while True:
            random_prob = np.random.rand()
            parent02_pool = [x for x in self.population if x.get_probability() <= random_prob]
            if parent02_pool:
                parent02 = parent02_pool[np.random.randint(len(parent02_pool))]
                if parent02 != parent01:
                    break
                else:
                    # Equal parents
                    continue
        if parent01 is not None and parent02 is not None:
            return parent01, parent02
        else:
            sys.exit(-1)

    def perform_mutation(self, child01_sequence, child02_sequence):
        """ performs mutation on a random byte for the child sequences """
        child01_sequence[randint(0, self.number_of_queens - 1)] = self.secure_random.choice(self.alpha_list)
        child02_sequence[randint(0, self.number_of_queens - 1)] = self.secure_random.choice(self.alpha_list)
        return child01_sequence, child02_sequence

    def perform_crossover(self, parent01, parent02):
        cross_over_point = randint(0, self.number_of_queens - 1)

        child01_sequence = list(parent01.get_sequence()[:cross_over_point]) + list(parent02.get_sequence()[cross_over_point:])
        child02_sequence = list(parent02.get_sequence()[:cross_over_point]) + list(parent01.get_sequence()[cross_over_point:])

        if uniform(0, 1) < self.mutation:
            child01_sequence, child02_sequence = self.perform_mutation(child01_sequence, child02_sequence)

        # Add both the children to the population
        for child_seq in [child01_sequence, child02_sequence]:
            child = ChessBoard()
            child.sequence = np.array(child_seq)
            child.set_fitness(self.calculate_individual_fitness(child.sequence))

            # Replace the item with 'minimum fitness' with the newly created child
            population_fitness = [x.fitness for x in self.population]
            self.population[population_fitness.index(min(population_fitness))] = child

    def solve(self):
        globals()
        try:
            self.generate_population()

            for loop in range(MAX_ITER):

                population_fitness = [x.get_fitness() for x in self.population]
                if self.stop_fitness in population_fitness:
                    print(self.population[population_fitness.index(max(population_fitness))].get_sequence())
                    break

                parent01, parent02 = self.get_parent()
                self.perform_crossover(parent01, parent02)

                self.calculate_probability()

                print('Iteration ', f'{loop:06}', ' ', ' Average Fitness Score ',
                      "{:.3f}".format(np.mean([x.get_fitness() for x in self.population])))

        except Exception as e:
            print('Error!!! Terminating script, Error description: {}'.format(e))


def main():
    solve_n_queens = SolveNQueens()

    solve_n_queens.set_number_of_queens(8)
    solve_n_queens.set_population_size(150)
    solve_n_queens.set_mutation(0.03)

    solve_n_queens.solve()


main()
