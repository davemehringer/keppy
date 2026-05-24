"""
Physical and mathematical constants used throughout keppy.

All values are in SI units unless otherwise noted.
"""

# Gravitational constant (m^3 kg^-1 s^-2)
G: float = 6.674_30e-11

# Astronomical unit in meters
AU: float = 1.495_978_707e11

# Julian day in seconds
DAY: float = 86_400.0

# Julian year in seconds
YEAR: float = 365.25 * DAY

# Solar gravitational parameter mu = G * M_sun  (m^3 s^-2)
MU_SUN: float = 1.327_124_400_18e20

# Speed of light (m/s)
C_LIGHT: float = 299_792_458.0
