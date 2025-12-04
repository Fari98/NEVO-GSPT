from main import _run, loaders, seeds, metrics, resnet_versions, sizes
from joblib import Parallel, delayed


for loader in loaders:
    for metric in metrics:
        for version in resnet_versions:
            for size in sizes:

                _ = Parallel(n_jobs=-1)(
                    delayed(_run)(
                        seed,
                        loader,
                        version,
                        metric,
                        size
                    ) for seed in range(seeds)
                )

