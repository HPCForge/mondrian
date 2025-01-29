import numpy as np
import torch

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
    # and a few are very high (>90) near the heater. 
    # having the model predict sqrt(temp) - sqrt(50) brings things closer to zero and
    # helps with the skew
    return np.sqrt(data) - np.sqrt(50)


def normalize_dfun(data):
    # the dfun in the liquid can basically be the domain size. I.e, around -12.
    # in vapor, the dfun is basically never larger than 1. 
    # Predicting the sqrt of the dfun makes it a little smaller and easier to manage.
    liquid_mask = data < 0
    data[liquid_mask] = -np.sqrt(abs(data[liquid_mask]))
    return data


def unnormalize_velx(data):
    return data / 2


def unnormalize_vely(data):
    return data / 2


def unnormalize_temperature(data):
    return (data + np.sqrt(50)) ** 2


def unnormalize_dfun(data):
    liquid_mask = data < 0
    data[liquid_mask] = -(data[liquid_mask] ** 2)
    return data


def unnormalize_data(data):
    is_tensor = torch.is_tensor(data)
    if is_tensor:
        arr = data.detach().cpu().numpy()
    else:
        arr = data
        
    s = arr.shape[1] // 4
    velx = unnormalize_velx(arr[:, :s])
    vely = unnormalize_vely(arr[:, s:2*s])
    temp = unnormalize_temperature(arr[:, 2*s:3*s])
    dfun = unnormalize_dfun(arr[:, 3*s:])

    normalized = np.concatenate((velx, vely, temp, dfun), axis=1)

    if is_tensor:
        return torch.from_numpy(normalized).to(data)
    return normalized