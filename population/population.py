from sklearn.metrics import root_mean_squared_error
import numpy as np

class Population:

    def __init__(self, individuals):
        self.individuals = individuals
        self.size = len(individuals)

        self.parameters = sum([ind.size for ind in self.individuals])

    def calculate_semantics(self, X, test=False):

        [ind.calculate_semantics(X, test) for ind in self.individuals]

    def evaluate(self, y, metric = root_mean_squared_error, test = False):

        [ind.evaluate(y, metric, test) for ind in self.individuals]

        self.fitnesses = [ind.fitness for ind in self.individuals]

    def find_elite(self, minimization = True):

        if minimization:
            self.elite = self.individuals[np.argmin(self.fitnesses)]
        else:
            self.elite = self.individuals[np.argmax(self.fitnesses)]

