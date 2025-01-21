from mondrian.dataset.bubbleml.constants import (
    normalize_velx,
    normalize_vely,
    normalize_temperature,
    unnormalize_velx,
    unnormalize_vely,
    unnormalize_temperature
)

def test_normalization():
    assert abs(unnormalize_velx(normalize_velx(10)) - 10) < 1e-3
    assert abs(unnormalize_vely(normalize_vely(10)) - 10) < 1e-3
    assert abs(unnormalize_temperature(normalize_temperature(10)) - 10) < 1e-3