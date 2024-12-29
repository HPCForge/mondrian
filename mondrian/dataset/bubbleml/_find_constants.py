r"""
This is just a script to find means and extents for each variable in the training dataset.
"""

import h5py
from dataclasses import dataclass, asdict


@dataclass
class Metrics:
    mean: float
    abs_max: float
    count: int

    def __str__(self):
        return f"MEAN={self.mean}, ABS_MAX={self.abs_max}"

    def __dict__(self):
        return asdict(self)


def update_metrics(cur, var_count, var_mean, var_abs_max):
    # probably not what I want... This is just adding everything up...,
    # but it also probably doesn't matter if the mean isn't very accurate.
    cur.mean = (cur.mean * cur.count + var_mean * var_count) / (cur.count + var_count)
    cur.count = cur.count + var_count
    cur.abs_max = max(cur.abs_max, var_abs_max)


EXCLUDE = ["int_runtime_params", "real_runtime_params"]


def print_dict(d):
    for key, value in d.items():
        print(f"{key}: {value}")


def main():
    train_data_path = "/pub/afeeney/Level-Set-Study/simulation/SubcooledBoiling/Train/train_fix_fix.hdf5"
    variable_metrics = {}
    with h5py.File(train_data_path, "r") as handle:
        for grp_name in handle.keys():
            print(f"Processing {grp_name}")
            grp = handle[grp_name]
            for var in grp.keys():
                print(f"Processing {var}")
                if var in EXCLUDE:
                    continue
                if var not in variable_metrics:
                    variable_metrics[var] = Metrics(0, 0, 0)

                data = grp[var][:]
                var_count = data.size
                var_mean = data.mean()
                var_abs_max = abs(data).max()
                update_metrics(variable_metrics[var], var_count, var_mean, var_abs_max)

    print_dict(variable_metrics)


if __name__ == "__main__":
    main()
