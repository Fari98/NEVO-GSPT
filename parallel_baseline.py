from baseline import _run, loaders, seeds
from joblib import Parallel, delayed


for i, loader in enumerate(loaders):

    _ = Parallel(n_jobs=3)(
        delayed(_run)(
            seed,
            loader,
            i
        ) for seed in range(seeds)
    )

