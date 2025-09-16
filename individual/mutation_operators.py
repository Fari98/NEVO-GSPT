import random

from block.block import Block
from block.utils import create_random_block
from utils.utils import uniform_random_step_generator
from individual.individual import Individual
import torch



def inflate_mutation(X_train,  ms_generator = uniform_random_step_generator(0, 1), X_test = None, neuron_probability = 1):

    input_dim = X_train.shape[1]

    def mutator(individual):
        block = create_random_block(input_dim, individual.structure[0].layers_dim, neuron_probability=neuron_probability)
        block = Block(block, individual.hidden_inputs, ms_generator(), individual.hidden_test_inputs)


        test_semantics = torch.cat((individual.total_test_semantics,
                                    block.forward(X_test, test=True)), dim = 1) if X_test is not None else None

        return Individual( individual.structure+[block],
                           torch.cat((individual.total_train_semantics, block.forward(X_train)), dim = 1),
                           total_test_semantics=test_semantics,
                           hidden_inputs=individual.hidden_inputs,
                           hidden_test_inputs=individual.hidden_test_inputs)

    return mutator

def deflate_mutation(individual):

    deflate_idx = random.randint(1, individual.size-2) #does not make sense to remove the last layer since that individual is already in the population

    test_semantics = torch.cat((individual.total_test_semantics[:, :deflate_idx],
                                  individual.total_test_semantics[:, deflate_idx+1:]), dim = 1) if (
                                    individual.total_test_semantics is not None) else None

    return Individual( structure = [*individual.structure[:deflate_idx], *individual.structure[deflate_idx+1:]],
                       total_train_semantics=torch.cat((individual.total_train_semantics[:, :deflate_idx],
                                  individual.total_train_semantics[:, deflate_idx+1:]), dim = 1),
                       total_test_semantics= test_semantics,
                       hidden_inputs=individual.hidden_inputs,
                       hidden_test_inputs = individual.hidden_test_inputs
                        )