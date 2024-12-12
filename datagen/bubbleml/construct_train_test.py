r"""
This script assumes that BubbleML simulations already exist.
It just prepares things for training, it does not run new
simulations.
There are several things it does:
    1. coarsens the spatial resolution for training
    2. removes any simulations with NaN values occuring after the steady time
    3. removes the "unsteady" time (first few hundred timesteps)
    4. Normalizes each field to [-1, 1]
    5. Redimensionalizes temp. The simulations normalize the temperature field by
       the wall temperature, so we rescale the temp field by the wall temp and THEN
       apply normalization
"""

import h5py
import numpy as np
import glob
import torch
import torch.nn.functional as F
import re

STEADY_TIME = 300

TEMPERATURE = "temperature"
PRESSURE = "pressure"
X_VELOCITY = "velx"
Y_VELOCITY = "vely"
DISTANCE_FUNC = "dfun"

FIELDS = [TEMPERATURE, PRESSURE, X_VELOCITY, Y_VELOCITY, DISTANCE_FUNC]

root = "/share/crsp/lab/ai4ts/share/simul_ts_0.1/PoolBoiling-SubCooled-FC72-2D-0.1/"

train_sims = [
    "Twall-102.hdf5",
    "Twall-106.hdf5",
    "Twall-110.hdf5",
    "Twall-81.hdf5",
    "Twall-88.hdf5",
    "Twall-92.hdf5",
    "Twall-97.hdf5",
    "Twall-103.hdf5",
    "Twall-108.hdf5",
    "Twall-79.hdf5",
    "Twall-98.hdf5",
]

test_sims = [
    "Twall-100.hdf5",
    "Twall-95.hdf5",
    "Twall-90.hdf5",
    "Twall-85.hdf5",
]


def coarsen_field(field, down_size):
    # interpolate expects a "batch" dimension
    dim = field.unsqueeze(0)
    coarsened = F.interpolate(dim, (down_size, down_size), mode="bilinear")
    return coarsened.squeeze(0)


def extents(field):
    return field.min(), field.max()


def get_extents(filenames):
    global_extents = dict([(field, (float("inf"), float("-inf"))) for field in FIELDS])
    for filename in filenames:
        print(f"getting extents {filename}")
        with h5py.File(f"{root}/{filename}", "r") as sim_handle:
            for field in FIELDS:
                # field_name: (min, max)
                global_extents[field] = (
                    min(global_extents[field][0], sim_handle[field][:].min()),
                    max(global_extents[field][1], sim_handle[field][:].max()),
                )
    return global_extents


def normalize(field, min, max):
    r"""
    Normalize the field to [-1, 1]
    """
    return 2 * ((field - min) / (max - min)) - 1


def build_train_set(filenames, extents, down_size=64):
    r"""
    The train dataset includes full simulations that
    are coarsened to be 64x64. Training is just done
    by using different time windows of these simulations
    Args:
        filenames: a list of paths to the simulation files
        extents: a list of dict of the training sets min/max for each field
        down_size: the target downsampling resolution
    """
    with h5py.File("bubbleml_train.hdf5", "w") as data_handle:
        for filename in filenames:
            p = re.compile("Twall-([0-9]+).hdf5")
            group_name = p.match(filename).group(1)
            wall_temp = float(group_name)
            grp = data_handle.create_group(f"{group_name}")
            with h5py.File(f"{root}/{filename}", "r") as sim_handle:
                # Do not create a dataset for the group if there are NaN values
                # occuring after the steady_time
                has_nan = False
                for field in FIELDS:
                    steady = sim_handle[field][STEADY_TIME:]
                    has_nan = np.any(np.isnan(steady))
                if has_nan:
                    print(f"sim {group_name} has nan in {field}")
                else:
                    for field in FIELDS:
                        data = torch.from_numpy(sim_handle[field][STEADY_TIME:])
                        lo, hi = extents[field]

                        if field == TEMPERATURE:
                            data *= wall_temp
                            lo *= wall_temp
                            hi *= wall_temp

                        normalized = normalize(data, lo, hi)
                        assert normalized.min() >= -1
                        assert normalized.max() <= 1
                        coarse = coarsen_field(normalized, down_size)
                        assert coarse.size(1) == down_size
                        assert coarse.size(2) == down_size
                        dataset = grp.create_dataset(field, data=coarse.numpy())
                        dataset.attrs["wall-temperature"] = wall_temp


def build_test_set(filenames, extents):
    r"""
    The test set is a collection of slices from a handful of simulations.
    These will have overlap than the training samples, just to avoid
    a lot of very similar samples.
    """
    with h5py.File("bubbleml_test.hdf5", "w") as data_handle:
        for filename in filenames:
            p = re.compile("Twall-([0-9]+).hdf5")
            group_name = p.match(filename).group(1)
            wall_temp = float(group_name)
            grp = data_handle.create_group(f"{group_name}")
            with h5py.File(f"{root}/{filename}", "r") as sim_handle:
                has_nan = False
                for field in FIELDS:
                    has_nan = np.any(np.isnan(sim_handle[field][STEADY_TIME:]))
                if has_nan:
                    print(f"sim {group_name} has nan in {field}")
                else:
                    for field in FIELDS:
                        data = sim_handle[field][STEADY_TIME:]
                        lo, hi = extents[field]

                        if field == TEMPERATURE:
                            data *= wall_temp
                            lo *= wall_temp
                            hi *= wall_temp
                        normalized = normalize(data, lo, hi)
                        dataset = grp.create_dataset(field, data=normalized)
                        dataset.attrs["wall-temperature"] = wall_temp


if __name__ == "__main__":
    extents = get_extents(train_sims)
    build_train_set(train_sims, extents)
    build_test_set(test_sims, extents)
    print("done")
