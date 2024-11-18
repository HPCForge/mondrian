r"""
Based on climatelearn's forecasting experiments. Use the listed era5 variables from
three past steps at interval of six hours to predict a future timestep.
"""

from climate_learn.data.processing.era5_constants import (
    PRESSURE_LEVEL_VARS,
    DEFAULT_PRESSURE_LEVELS,
)

from .itermodule import IterDataModule

ERA5_VARIABLES = [
    'geopotential',
    'temperature',
    'u_component_of_wind',
    'v_component_of_wind',
    'relative_humidity',
    'specific_humidity',
    '2m_temperature',
    '10m_u_component_of_wind',
    '10m_v_component_of_wind',
    'toa_incident_solar_radiation',
    'land_sea_mask',
    'orography',
    'lattitude'
]

PRED_RANGE_OPTIONS = [6, 24, 72, 120, 240]

def get_era5_dataloaders(data_path,
                         pred_range,
                         batch_size,
                         num_workers,
                         pin_memory):
  assert pred_range in PRED_RANGE_OPTIONS
  
  # Use of the ERA5 variables, and five pressure levels.
  in_vars = []
  for var in ERA5_VARIABLES:
    if var in PRESSURE_LEVEL_VARS:
        for level in DEFAULT_PRESSURE_LEVELS:
            in_vars.append(var + '_' + str(level))
    else:
        in_vars.append(var)
        
  out_vars = [
    '2m_temperature', 'geopotential_500', 'temperature_850'
  ]
  
  dm = IterDataModule(
        'direct-forecasting',
        # input/output paths are the same, since we are
        # just trying to predict future timesteps in the same file.
        data_path,
        data_path,
        in_vars,
        out_vars,
        src='era5',
        history=3,
        window=6,
        pred_range=pred_range,
        # train on intervals of 6 hours, rather than using 
        # every hour in the dataset
        subsample=6,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
  dm.setup()
  return dm