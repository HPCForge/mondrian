# The liquid's temperature
BULK_TEMP = 50

# The liquid's boiling point
SATURATION_TEMP = 58


# the constants are found by running _find_constants on the training script.
# the dataset is large enough that it's annoying to find these constants for each
# training run. I also want the data to be saved unnormalized, so it's possible
# to try different methods for normalizing.
def normalize_velx(data):
    VELX_MEAN = 0.010963355915906966
    VELX_ABS_MAX = 15.963545954835148
    return (data - VELX_MEAN) / (VELX_ABS_MAX - VELX_MEAN)


def normalize_vely(data):
    VELY_MEAN = -0.02075080353091658
    VELY_ABS_MAX = 18.058848630729504
    return (data - VELY_MEAN) / (VELY_ABS_MAX - VELY_MEAN)


def normalize_temperature(data):
    TEMPERATURE_MEAN = 51.31273659095467
    TEMPERATURE_ABS_MAX = 96.17889078668429
    return (data - TEMPERATURE_MEAN) / (TEMPERATURE_ABS_MAX - TEMPERATURE_MEAN)


def normalize_mass_flux(data):
    MASS_FLUX_MEAN = 4.987113324387662e-06
    MASS_FLUX_ABS_MAX = 0.05450146197563288
    return (data - MASS_FLUX_MEAN) / (MASS_FLUX_ABS_MAX - MASS_FLUX_MEAN)


def normalize_dfun(data):
    DFUN_MEAN = -2.791855041377248
    DFUN_ABS_MAX = 12.419516294521564
    return (data - DFUN_MEAN) / (DFUN_ABS_MAX - DFUN_MEAN)
