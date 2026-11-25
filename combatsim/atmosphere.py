"""Shared atmosphere model for Python and Numba motion implementations."""
import math

from numba.extending import register_jitable


@register_jitable
def density(altitude):
    """Dry troposphere with an isothermal continuation above 11 km (kg/m^3)."""
    h = max(0.0, altitude)
    temperature = 288.15 - .0065*min(h, 11000.0)
    pressure = 101325.0*(temperature/288.15)**(9.80665/(287.05*.0065))
    if h > 11000.0:
        pressure *= math.exp(-9.80665*(h-11000.0)/(287.05*temperature))
    return pressure/(287.05*temperature)


@register_jitable
def thrust_fraction(air_density, exponent):
    """Toy air-breathing engine lapse; exponent zero retains constant thrust."""
    return min(1.0, max(0.0, air_density/density(0.0)))**exponent
