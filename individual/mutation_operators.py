import random

from block.block import Block
from block.utils import create_random_block
from utils.utils import uniform_random_step_generator, normalize_block_semantics
from individual.individual import Individual
import torch



def inflate_mutation(X_train,  ms_generator = uniform_random_step_generator(0, 1), X_test = None, neuron_probability = 1, n_classes = 1):

    input_dim = X_train.shape[1]

    def mutator(individual):
        block = create_random_block(input_dim, individual.layers_dim,
                                    neuron_probability=neuron_probability, n_classes=n_classes)
        block = Block(block, individual.hidden_inputs, ms_generator(), individual.hidden_test_inputs)

        # Get new block's raw outputs
        new_block_train_sem = block.forward(X_train)
        new_block_test_sem = block.forward(X_test, test=True) if X_test is not None else None

        # Normalize new block semantics if requested
        # The new block is at index = individual.length (number of existing blocks)
        if individual.normalize_semantics and n_classes > 1:
            new_block_train_sem = normalize_block_semantics(
                new_block_train_sem,
                block_idx=individual.length,
                n_classes=n_classes,
                multiclass=True
            )
            if new_block_test_sem is not None:
                new_block_test_sem = normalize_block_semantics(
                    new_block_test_sem,
                    block_idx=individual.length,
                    n_classes=n_classes,
                    multiclass=True
                )

        # Concatenate normalized semantics
        train_semantics = torch.cat((individual.total_train_semantics, new_block_train_sem), dim=1)
        test_semantics = torch.cat((individual.total_test_semantics, new_block_test_sem), dim=1) if X_test is not None else None

        return Individual( individual.structure+[block],
                           train_semantics,
                           total_test_semantics=test_semantics,
                           hidden_inputs=individual.hidden_inputs,
                           hidden_test_inputs=individual.hidden_test_inputs,
                           multiclass=n_classes>1,
                           n_classes=n_classes,
                           layers_dim=individual.layers_dim,
                           normalize_semantics=individual.normalize_semantics)

    return mutator

def deflate_mutation(individual, multiclass = False, n_classes = 1):

    # For multiclass, we need to remove n_classes columns at a time to maintain proper dimensions
    if multiclass and n_classes > 1:
        # Calculate number of blocks
        total_features = individual.total_train_semantics.shape[1]
        n_blocks = total_features // n_classes

        if n_blocks <= 1:
            # Can't deflate if only 1 block
            return individual

        # Select a random block to remove (0-indexed)
        block_idx = random.randint(1, n_blocks - 1)

        # Calculate the column indices to remove
        start_col = block_idx * n_classes
        end_col = start_col + n_classes

        # Remove the block's columns from semantics
        train_semantics_parts = [
            individual.total_train_semantics[:, :start_col],
            individual.total_train_semantics[:, end_col:]
        ]
        train_semantics = torch.cat([p for p in train_semantics_parts if p.shape[1] > 0], dim=1)

        if individual.total_test_semantics is not None:
            test_semantics_parts = [
                individual.total_test_semantics[:, :start_col],
                individual.total_test_semantics[:, end_col:]
            ]
            test_semantics = torch.cat([p for p in test_semantics_parts if p.shape[1] > 0], dim=1)
        else:
            test_semantics = None

        # Remove the block from structure
        new_structure = [*individual.structure[:block_idx], *individual.structure[block_idx+1:]]

    else:
        # Original logic for regression/binary classification
        deflate_idx = random.randint(1, individual.size-2)

        test_semantics = torch.cat((individual.total_test_semantics[:, :deflate_idx],
                                      individual.total_test_semantics[:, deflate_idx+1:]), dim = 1) if (
                                        individual.total_test_semantics is not None) else None

        train_semantics = torch.cat((individual.total_train_semantics[:, :deflate_idx],
                                      individual.total_train_semantics[:, deflate_idx+1:]), dim = 1)

        new_structure = [*individual.structure[:deflate_idx], *individual.structure[deflate_idx+1:]]

    return Individual( structure = new_structure,
                       total_train_semantics=train_semantics,
                       total_test_semantics= test_semantics,
                       hidden_inputs=individual.hidden_inputs,
                       hidden_test_inputs = individual.hidden_test_inputs,
                       multiclass=multiclass,
                       n_classes=n_classes,
                       layers_dim=individual.layers_dim,
                       normalize_semantics=individual.normalize_semantics
                        )