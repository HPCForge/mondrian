import numpy as np

# The liquid's temperature for subcooled boiling
SUBCOOLED_BULK_TEMP = 50

# The liquid's boiling point
SATURATION_TEMP = 58


# the constants are found by running _find_constants on the training script.
# the dataset is large enough that it's annoying to find these constants for each
# training run. I also want the data to be saved unnormalized, so it's possible
# to try different methods for normalizing.


# NOTE: velocities are already very close to zero mean and look gaussian.
# y-velocities may be slightly biased towards being positive, since
# heater makes things go up! Dividing by 0.5 makes things ~unit variance
def normalize_velx(data):
    return data / 0.5


def normalize_vely(data):
    return data / 0.5


def normalize_temperature(data):
    # NOTE: the distribution of temperatures is horribly skewed. So normalizing to 
    # a gaussian isn't really possible. 
    # The huge majority of temperature values are around the bulk temperature (~50),
    # and a few are very high (>90) near the heater temperature. 
    # these numbers are just chosen manually based on a little analysis that at
    # least get it around zero mean and np.var ~= 1.
    return (data - 52) / 2.8
    

def normalize_dfun(data):
    DFUN_MAX = 1.5645649406815259
    DFUN_MIN = -12.419516294521564
    x = (data - DFUN_MIN) / (DFUN_MAX - DFUN_MIN)
    return 2 * x - 1
