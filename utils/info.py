import os
import csv
from filelock import FileLock

def get_log_info(optimizer, log):
    base_log = [
        optimizer.population.elite.fitness,
        optimizer.population.elite.test_fitness,
        optimizer.population.elite.size,
        optimizer.population.elite.length,
        len([x for x in list(optimizer.population.elite.structure[0].layers.keys()) if 'hidden' in x])
    ]

    if log == 1:
        return base_log
    # elif log == 2:
    #
    #     return base_log + [[individual.full_perf1 for individual in optimizer.elites]]
    #
    # else:
    #     return base_log + [[individual.representations for individual in optimizer.elites]]
    else:
        raise Exception("Invalid log level")



def base_logger(
    path: str,
    row : list
) -> None:
    """
    Logs information into a CSV file.

    Parameters
    ----------
    path : str
        Path to the CSV file.
    generation : int
        Current generation number.
    timing : float
        Time taken for the process.
    run_info : list, optional
        Information about the run. Defaults to None.
    seed : int, optional
        The seed used in random, numpy, and torch libraries. Defaults to 0.

    Returns
    -------
    None
    """
    if not os.path.isdir(os.path.dirname(path)):
        os.mkdir(os.path.dirname(path))

    lock_path = path + '.lock'
    with FileLock(lock_path):
        with open(path, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(row)

def logger(
        path: str,
        generation: int,
        timing: float,
        run_info: list = None,
        seed: int = 0,
) -> None:
    """
    Logs information into a CSV file.

    Parameters
    ----------
    path : str
        Path to the CSV file.
    generation : int
        Current generation number.
    timing : float
        Time taken for the process.
    run_info : list, optional
        Information about the run. Defaults to None.
    seed : int, optional
        The seed used in random, numpy, and torch libraries. Defaults to 0.

    Returns
    -------
    None
    """

    if not os.path.isdir(os.path.dirname(path)):
        os.mkdir(os.path.dirname(path))

    lock_path = path + '.lock'
    with FileLock(lock_path):
        with open(path, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(run_info + [seed, generation, timing])


def verbose_reporter(
        dataset, generation, train_fitness, test_fitness, params, timing
):
    """
    Prints a formatted report of generation, fitness values, timing, and node count.

    Parameters
    ----------
    generation : int
        Current generation number.
    train_fitness : float
        Population's train_fitness fitness value.
    test_fitness : float
        Population's obj2 fitness value.
    timing : float
        Time taken for the process.


    Returns
    -------
    None
        Outputs a formatted report to the console.
    """
    digits_dataset = len(str(dataset))
    digits_generation = len(str(generation))
    digits_val_fit = len(str(float(train_fitness)))
    digits_size = len(str(float(params)))
    if test_fitness is not None:
        digits_test_fit = len(str(float(test_fitness)))
        test_text_init = (
                "|"
                + " " * 3
                + str(float(test_fitness))
                + " " * (23 - digits_test_fit)
                + "|"
        )
        test_text = (
                " " * 3 + str(float(test_fitness)) + " " * (23 - digits_test_fit) + "|"
        )
    else:
        digits_test_fit = 4
        test_text_init = "|" + " " * 3 + "None" + " " * (23 - digits_test_fit) + "|"
        test_text = " " * 3 + "None" + " " * (23 - digits_test_fit) + "|"
    digits_timing = len(str(timing))

    if generation == 0:
        print("Verbose Reporter")
        print(
            "----------------------------------------------------------------------------------------------------------------------"
        )
        print(
            "|         Dataset         |  Generation  |        Train        |         Test        |         Params        |        "
            "Timing          |"
        )
        print(
            "----------------------------------------------------------------------------------------------------------------------"
        )
        print(
            "|"
            + " " * 5
            + str(dataset)
            + " " * (20 - digits_dataset)
            + "|"
            + " " * 7
            + str(generation)
            + " " * (7 - digits_generation)
            + "|"
            + " " * 3
            + str(float(train_fitness))
            + " " * (20 - digits_val_fit)
            + test_text_init
            + " " * 3
            + str(params)
            + " " * (21 - digits_size)
            + "|"
            + " " * 3
            + str(timing)
            + " " * (21 - digits_timing)
            + "|"
        )
    else:
        print(
            "|"
            + " " * 5
            + str(dataset)
            + " " * (20 - digits_dataset)
            + "|"
            + " " * 7
            + str(generation)
            + " " * (7 - digits_generation)
            + "|"
            + " " * 3
            + str(float(train_fitness))
            + " " * (20 - digits_val_fit)
            + "|"
            + test_text
            + " " * 3
            + str(params)
            + " " * (21 - digits_size)
            + "|"
            + " " * 3
            + str(timing)
            + " " * (21 - digits_timing)
            + "|"
        )