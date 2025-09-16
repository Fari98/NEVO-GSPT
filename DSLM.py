import random
import time
import numpy as np
import torch
from individual.individual import Individual

from utils.info import logger, verbose_reporter, get_log_info

from population.population import Population
from sklearn.metrics import root_mean_squared_error


class DSLM:
    def __init__(
        self,
        initializer,
        selector,
        inflate_mutator,
        deflate_mutator,
        crossover=None,
        p_m = 1,
        p_im=0.3,
        p_dm = 0.7,
        p_xo=0,
        pop_size=100,
        seed=0,
    ):

        self.selector = selector
        self.p_m = p_m
        self.p_im = p_im
        self.p_dm = p_dm
        self.crossover = crossover
        self.inflate_mutator = inflate_mutator
        self.deflate_mutator = deflate_mutator
        self.p_xo = p_xo
        self.initializer = initializer
        self.pop_size = pop_size
        self.seed = seed

    def solve(
        self,
        X_train,
        y_train,
        X_test = None,
        y_test = None,
        metric = root_mean_squared_error,
        max_depth = None,
        generations=20,
        elitism=True,
        dataset_name=None,
        log=0,
        log_path = None,
        verbose=0,
        n_jobs = 1
    ):
        """
        """

        self.log = log
        # setting the seeds
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        start = time.time()

        # Initialize the population
        self.population = self.initializer(self.pop_size, n_jobs)

        self.population = Population([Individual([ind], ind.forward(X_train), ind.forward(X_test)) for ind in self.population])

        # evaluating the intial population
        self.population.calculate_semantics(X_train)
        self.population.evaluate(y_train, metric)
        [ind.create_hidden_input(X_train) for ind in self.population.individuals]

        if X_test is not None:
            self.population.calculate_semantics(X_test, test = True)
            self.population.evaluate(y_test, metric, test = True)
            [ind.create_hidden_input(X_test, test = True) for ind in self.population.individuals]

        end = time.time()

        if elitism:
            self.population.find_elite()

        # logging the results if the log level is not 0

        timing = end-start

        if self.log != 0:

            logger(log_path,
                   0,
                   timing,
                   [dataset_name] + get_log_info(self, log),
                   self.seed)

        # displaying the results on console if verbose level is not 0
        if verbose != 0:
            verbose_reporter(
                dataset_name,
                0,
                self.population.elite.fitness,
                self.population.elite.test_fitness,
                self.population.elite.size,
                timing
            )

        # EVOLUTIONARY PROCESS
        for generation in range(1, generations + 1):

            start = time.time()

            if elitism:
                offs_pop = [self.population.elite]
            else:
                offs_pop = []
            while len(offs_pop) < self.population.size:

                if random.random() < self.p_m:

                    parent = self.selector(self.population)

                    if random.random() < self.p_im:
                        offspring = self.inflate_mutator(parent)

                    else:
                        offspring = self.deflate_mutator(parent)

                    offs_pop.append(offspring)

                else:

                    p1, p2 = self.selector(self.population), self.selector(self.population)
                    offs1, offs2 = self.crossover(p1, p2, max_depth)

                    offs_pop.extend([offs1, offs2])

            offs_pop = offs_pop[:self.population.size]
            offs_pop = Population(offs_pop)
            # replacing the population with the offspring population (P = P')
            self.population = offs_pop


            self.population.calculate_semantics(X_train)
            self.population.evaluate(y_train, metric)

            if X_test is not None:
                self.population.calculate_semantics(X_test, test=True)
                self.population.evaluate(y_test, metric, test=True)

            # getting the new elite(s)
            if elitism:
               self.population.find_elite()

            end = time.time()

            timing = end - start

            if log != 0:

                logger(log_path,
                       generation,
                       timing,
                       [dataset_name] + get_log_info(self, self.log),
                       self.seed)

            # displaying the results on console if verbose level is not 0
            if verbose != 0:
                verbose_reporter(
                    dataset_name,
                    generation,
                    self.population.elite.fitness,
                    self.population.elite.test_fitness,
                    self.population.elite.size,
                    timing
                )