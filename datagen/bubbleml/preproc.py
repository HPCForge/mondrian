import argparse
import h5py
import numpy as np
import re

TEMPERATURE = 'temperature'
PRESSURE = 'pressure'
X_VELOCITY = 'velx'
Y_VELOCITY = 'vely'
DISTANCE_FUNC = 'dfun'
MASS_FLUX = 'mass_flux'

# Fields should be normalized and potentially passed into model
FIELDS = [
    TEMPERATURE,
    PRESSURE,
    X_VELOCITY,
    Y_VELOCITY,
    DISTANCE_FUNC,
    MASS_FLUX
]

NUCLEATION_SITES_X = 'nucleation_sites_x'
NUCLEATION_DFUN = 'nucleation_dfun'
X = 'x'
Y = 'y'

# assuming subcooled boiling, liquid has temperature 50
BULK_TEMP = 50

def main(args):
    extents = preproc_train(args.read_train_file, args.write_train_file)
    preproc(args.read_test_file, args.write_test_file, extents)

def get_wall_temp(key):
    r"""
    Read the wall temperature from hdf5 group key
    """
    nums = re.compile(r'\d+')
    wall_temp = nums.search(key).group(0)
    return int(wall_temp)

def nucleation_dfun(sites_x, x, y):
    # nucleation sites are set 1e-13 above heater surface
    # in simulation files.
    sites_y = np.full_like(sites_x, 1e-13)
    sites = np.stack((sites_x, sites_y), axis=1)

    # dist[i, j] is the distance to the closest nucleation site
    min_dist = np.full_like(x, 1e10)
    # from each grid cell, compute distance to nuclation site
    for idx in range(sites.shape[0]):
        coord = sites[idx]
        dist = np.sqrt((x - coord[0]) ** 2 + (y - coord[1]) ** 2)
        min_dist = np.minimum(min_dist, dist)
    return min_dist

def get_extents(handle):
    r"""
    Get the min and max, across all groups, of each field
    """
    global_extents = dict([(field, (float('inf'), float('-inf'))) for field in FIELDS])
    for k in handle.keys():
        grp = handle[k]
        for field in FIELDS:
            # field_name: (min, max)
            global_extents[field] = (
                min(global_extents[field][0], grp[field][:].min()),
                max(global_extents[field][1], grp[field][:].max())
            )
    return global_extents

def normalize(field, min, max):
    r"""
    Normalize the field to [-1, 1]
    """
    return 2 * ((field - min) / (max - min)) - 1

def preproc(read_path, write_path, extents):
    with h5py.File(read_path, 'r') as read_handle:
        with h5py.File(write_path, 'w') as write_handle:
            extents = get_extents(read_handle)
            for k in read_handle.keys():
                print(f'processing {k}')
                wall_temp = get_wall_temp(k)
                read_grp = read_handle[k]
                write_grp = write_handle.create_group(k)
                for field in FIELDS:
                    data = read_grp[field][:]
                    lo, hi = extents[field]
                    if field == TEMPERATURE:
                        data = (data * wall_temp) + BULK_TEMP
                        lo = (lo * wall_temp) + BULK_TEMP
                        hi = (hi * wall_temp) + BULK_TEMP

                    normalized = normalize(data, lo, hi) 
                    assert normalized.min() >= -1
                    assert normalized.max() <= 1
                    write_grp.create_dataset(field, data=np.squeeze(normalized))

                write_grp.attrs['physical_wall_temp'] = wall_temp

                #nuc_dfun = nucleation_dfun(read_grp[NUCLEATION_SITES_X],
                #                           read_grp[X][0, 0],
                #                           read_grp[Y][0, 0])
                #write_grp.create_dataset(NUCLEATION_DFUN, data=nuc_dfun)
                write_grp.create_dataset(NUCLEATION_SITES_X,
                                         data=read_grp[NUCLEATION_SITES_X][:])

    return extents

def preproc_train(read_path, write_path):
    with h5py.File(read_path, 'r') as read_handle:
        extents = get_extents(read_handle)
    preproc(read_path, write_path, extents)
    return extents

if __name__ == '__main__':
    # sanity checks
    assert get_wall_temp('Twall-97-1234') == 97

    parser = argparse.ArgumentParser()
    parser.add_argument('--read_train_file', type=str, required=True)
    parser.add_argument('--write_train_file', type=str, required=True)
    parser.add_argument('--read_test_file', type=str, required=True)
    parser.add_argument('--write_test_file', type=str, required=True)
    args = parser.parse_args()

    assert args.read_train_file != args.write_train_file
    assert args.read_test_file != args.write_test_file

    main(args)
