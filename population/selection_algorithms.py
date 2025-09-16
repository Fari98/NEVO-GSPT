import random
import numpy as np

def torunament_selection(pool_size):

    def ts(population):

        pool = random.choices(population.individuals, k=pool_size)
        return pool[np.argmin([ind.fitness for ind in pool])]

    return ts