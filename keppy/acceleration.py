"""
AccelerationCalculator — computes gravitational accelerations for an N-body system.

Improvements over C++ keplerpp:
  * The DistanceCalculator is not a separate class — NumPy broadcasting
    computes all pairwise displacements and distances in two vectorized lines,
    eliminating the need for pre-allocated distance matrices.
  * PairwiseAccelerationCalculator is fully vectorized: no Python-level loops
    over body pairs.
  * Zonal harmonic perturbations (J2–J10) are factored into a standalone
    function that is easy to test and extend.
  * The base class is a typing.Protocol so any callable object with the right
    signature works — no forced inheritance.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from keppy.constants import G


# ---------------------------------------------------------------------------
# Protocol (interface)
# ---------------------------------------------------------------------------

@runtime_checkable
class AccelerationCalculator(Protocol):
    """
    Interface for acceleration calculators.

    Any object that implements ``compute(positions, mus)`` satisfies this
    protocol and can be passed to an integrator.
    """

    def compute(
        self,
        positions: np.ndarray,
        mus: np.ndarray,
    ) -> np.ndarray:
        """
        Compute gravitational accelerations for all bodies.

        Parameters
        ----------
        positions : shape (n, 3), meters
        mus       : shape (n,),   gravitational parameters G·m (m³ s⁻²)

        Returns
        -------
        accelerations : shape (n, 3), m s⁻²
        """
        ...


# ---------------------------------------------------------------------------
# Pairwise Newtonian gravity (vectorized)
# ---------------------------------------------------------------------------

class PairwiseAccelerationCalculator:
    """
    Computes accelerations from pairwise Newtonian gravity.

    All O(n²) pair interactions are handled in two NumPy broadcasting steps,
    with no Python loop over pairs.

    Optionally applies zonal harmonic perturbations (J2–J10) for bodies that
    have ``j_coefficients`` and a ``center_body`` set.  The perturbation is
    only evaluated when the orbiting body is within ``j_radius_factor`` ×
    the perturbing body's radius (default 100, matching keplerpp).

    Parameters
    ----------
    bodies           : list of Body objects (used for J-perturbation metadata).
                       Pass an empty list (or None) for pure point-mass runs.
    j_radius_factor  : Multiplier on body radius to define the J-perturbation
                       cutoff distance.  Set to 0 to disable J terms entirely.
    """

    def __init__(self, bodies=None, j_radius_factor: float = 100.0) -> None:
        self._bodies = bodies or []
        self._j_radius_factor = j_radius_factor

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def compute(
        self,
        positions: np.ndarray,
        mus: np.ndarray,
    ) -> np.ndarray:
        """
        Return accelerations (m s⁻²) for all bodies, shape (n, 3).

        Parameters
        ----------
        positions : shape (n, 3), meters
        mus       : shape (n,),   m³ s⁻²
        """
        positions = np.asarray(positions, dtype=float)
        mus = np.asarray(mus, dtype=float)
        n = len(mus)

        # diff[i, j] = positions[j] - positions[i]  →  shape (n, n, 3)
        diff = positions[np.newaxis, :, :] - positions[:, np.newaxis, :]

        # Squared distances and distances, with diagonal set to 1 to avoid /0
        d2 = np.einsum("ijk,ijk->ij", diff, diff)       # (n, n)
        np.fill_diagonal(d2, 1.0)
        d = np.sqrt(d2)                                  # (n, n)

        # Acceleration from body j on body i:
        #   a_ij = mu_j * diff_ij / d_ij^3
        # shape: (n, n, 1) * (n, n, 3) / (n, n, 1)  →  (n, n, 3)
        d3 = d2 * d
        acc = (mus[np.newaxis, :, np.newaxis] * diff) / d3[:, :, np.newaxis]

        # Zero out self-interaction (i == j diagonal)
        mask = np.eye(n, dtype=bool)
        acc[mask] = 0.0

        # Sum contributions from all other bodies
        result = acc.sum(axis=1)   # (n, 3)

        # Add zonal harmonic perturbations where applicable
        if self._bodies and self._j_radius_factor > 0:
            result += self._j_perturbations(positions, d, d2)

        return result

    # ------------------------------------------------------------------
    # Zonal harmonic perturbations
    # ------------------------------------------------------------------

    def _j_perturbations(
        self,
        positions: np.ndarray,
        d: np.ndarray,
        d2: np.ndarray,
    ) -> np.ndarray:
        """
        Accumulate J2–J10 acceleration corrections, shape (n, 3).

        Only evaluated when:
          (a) the perturbing body has j_coefficients, and
          (b) the orbiting body has center_body pointing at the perturber, and
          (c) the distance is within j_radius_factor × perturber.radius.
        """
        n = len(self._bodies)
        result = np.zeros((n, 3))

        for j_idx, perturber in enumerate(self._bodies):
            if not perturber.j_coefficients or perturber.radius == 0:
                continue
            r_limit2 = (self._j_radius_factor * perturber.radius) ** 2

            for i_idx, orbiter in enumerate(self._bodies):
                if i_idx == j_idx:
                    continue
                if orbiter.center_body is None:
                    continue
                if orbiter.center_body.body_id != perturber.body_id:
                    continue
                if d2[i_idx, j_idx] >= r_limit2:
                    continue

                # Position of orbiter relative to perturber, in equatorial frame.
                # If a rotation matrix is available on the perturber use it;
                # otherwise assume ecliptic ≈ equatorial (identity rotation).
                diff_ec = positions[i_idx] - positions[j_idx]
                if perturber.rotation_matrix is not None:
                    diff_eq = perturber.rotation_matrix @ diff_ec
                else:
                    diff_eq = diff_ec

                dij = d[i_idx, j_idx]
                d2ij = d2[i_idx, j_idx]

                aj = _zonal_harmonic_acceleration(
                    diff_eq, dij, d2ij,
                    perturber.radius,
                    perturber.j_coefficients,
                )

                # Rotate result back to ecliptic frame if needed
                if perturber.rotation_matrix is not None:
                    aj = perturber.rotation_matrix.T @ aj

                result[i_idx] += aj

        return result


# ---------------------------------------------------------------------------
# Zonal harmonic acceleration (standalone, testable)
# ---------------------------------------------------------------------------

def _zonal_harmonic_acceleration(
    diff_eq: np.ndarray,
    d: float,
    d2: float,
    radius: float,
    j_coefficients: list[float],
) -> np.ndarray:
    """
    Acceleration perturbation (m s⁻²) from zonal harmonics J2–J10.

    Implements the same polynomial expansion as keplerpp's ``doJContrib``.
    All quantities must be in the body's equatorial frame.

    Parameters
    ----------
    diff_eq       : position of orbiter relative to perturber (equatorial), m
    d             : scalar distance |diff_eq|, m
    d2            : d², m²
    radius        : equatorial radius of the perturbing body, m
    j_coefficients: [J2, J3, J4, …]  (index 0 = J2)

    Returns
    -------
    aj : shape (3,), m s⁻²
    """
    x, y, z = diff_eq
    r_norm = d / radius          # dimensionless distance in body radii
    r2 = r_norm * r_norm

    d3 = d2 * d
    z_d = z / d
    z_d2 = z_d * z_d

    # --- J2 ---
    # c[2] in keplerpp = mu * J2 * R^2  (stored pre-multiplied)
    # Here we compute the coefficient on the fly from J2 directly.
    # The standard J2 acceleration formula (in body-equatorial frame):
    #   a_xy = mu*J2*R^2/d^5 * x * (5z²/d² - 1) * (-3/2)    [keplerpp sign]
    #   a_z  = mu*J2*R^2/d^5 * z * (5z²/d² - 3) * (-3/2)
    # keplerpp uses c[n] = mu * Jn * R^n, accessed as body.c[n].
    # Since we don't pre-multiply mu here, we return a unit-less shape
    # contribution that the caller scales by mu if needed.
    # HOWEVER — in keplerpp the full pairwise term already includes mu/d²,
    # and doJContrib adds a *correction* on top.  The correction uses
    # body.c[n] = mu * Jn * (R/d)^n / d^3 effectively.
    # We replicate that exactly below.

    nj = len(j_coefficients)     # j_coefficients[0] = J2
    aj = np.zeros(3)

    if nj < 1 or j_coefficients[0] == 0:
        # No J2 → nothing to compute
        return aj

    # keplerpp stores c[n] = mu_body * J_n * R^(n-0) pre-divided differently;
    # here we reproduce the math directly.
    # For J2:  cr = J2 / r^2   where r = d/R  →  cr = J2 * R^2 / d^2
    cr = j_coefficients[0] / r2
    _2_5_z_d2 = 2.5 * z_d2
    ff = cr * (-0.5 + _2_5_z_d2) / d3
    aj[0] = x * ff
    aj[1] = y * ff
    aj[2] = cr * z * (-1.5 + _2_5_z_d2) / d3

    if nj < 2:
        return aj

    d4 = d2 * d2
    z_d4 = z_d2 * z_d2

    # --- J3 ---
    if nj >= 2 and j_coefficients[1] != 0:
        r3 = r2 * r_norm
        cr = j_coefficients[1] / r3
        ff = 10 * z * cr * (-1.5 + 3.5 * z_d2) / d4
        aj[0] += x * ff
        aj[1] += y * ff
        aj[2] += cr * (3.0 - 30.0 * z_d2 + 35.0 * z_d4) / d2

    # --- J4 ---
    if nj >= 3 and j_coefficients[2] != 0:
        r4 = r2 * r2
        cr = j_coefficients[2] / r4
        ff = 6.0 * cr * (0.125 - 1.75 * z_d2 + 2.625 * z_d4) / d3
        aj[0] += ff * x
        aj[1] += ff * y
        aj[2] += cr * z * (3.75 - 17.5 * z_d2 + 15.75 * z_d4) / d3

    if nj < 4:
        return aj

    z_d6 = z_d2 * z_d4

    # --- J5 ---
    if nj >= 4 and j_coefficients[3] != 0:
        r4 = r2 * r2
        r5 = r4 * r_norm
        cr = j_coefficients[3] / r5
        ff = 56.0 * cr * (0.625 - 3.75 * z_d2 + 4.125 * z_d4) / d4
        aj[0] += ff * x * z
        aj[1] += ff * y * z
        aj[2] += cr * (-5.0 + 105.0 * z_d2 - 315.0 * z_d4 + 231.0 * z_d6) / d2

    # --- J6 ---
    if nj >= 5 and j_coefficients[4] != 0:
        r4 = r2 * r2
        r6 = r4 * r2
        cr = j_coefficients[4] / r6
        ff = 8.0 * cr * (
            -0.3125 + 8.4375 * z_d2 - 30.9375 * z_d4 + 26.8125 * z_d6
        ) / d3
        aj[0] += ff * x
        aj[1] += ff * y
        aj[2] += cr * z * (
            -17.5 + 157.5 * z_d2 - 346.5 * z_d4 + 214.5 * z_d6
        ) / d3

    if nj < 6:
        return aj

    z_d8 = z_d4 * z_d4

    # --- J7 ---
    if nj >= 6 and j_coefficients[5] != 0:
        r4 = r2 * r2
        r6 = r4 * r2
        r7 = r6 * r_norm
        cr = j_coefficients[5] / r7
        ff = 144 * cr * (
            -2.1875 + 24.0625 * z_d2 - 62.5625 * z_d4 + 44.6875 * z_d6
        ) / d4
        aj[0] += x * z * ff
        aj[1] += y * z * ff
        aj[2] += cr * (
            35.0 - 1260.0 * z_d2 + 6930.0 * z_d4
            - 12012.0 * z_d6 + 6435.0 * z_d8
        ) / d2

    # --- J8 ---
    if nj >= 7 and j_coefficients[6] != 0:
        r4 = r2 * r2
        r6 = r4 * r2
        r8 = r6 * r2
        cr = j_coefficients[6] / r8
        ff = 80 * cr * (
            0.0546875 - 2.40625 * z_d2 + 15.640625 * z_d4
            - 31.28125 * z_d6 + 18.9921875 * z_d8
        ) / d3
        aj[0] += ff * x
        aj[1] += ff * y
        aj[2] += cr * z * (
            39.375 - 577.5 * z_d2 + 2252.25 * z_d4
            - 3217.5 * z_d6 + 1519.375 * z_d8
        ) / d3

    if nj < 8:
        return aj

    z_d10 = z_d6 * z_d4

    # --- J9 ---
    if nj >= 8 and j_coefficients[7] != 0:
        r4 = r2 * r2
        r6 = r4 * r2
        r8 = r6 * r2
        r9 = r8 * r_norm
        cr = j_coefficients[7] / r9
        ff = 1408 * cr * (
            0.4921875 - 8.53125 * z_d2 + 38.390625 * z_d4
            - 62.15625 * z_d6 + 32.8046875 * z_d8
        ) / d4
        aj[0] += x * z * ff
        aj[1] += y * z * ff
        aj[2] += cr * (
            -63.0 + 3465.0 * z_d2 - 30030.0 * z_d4
            + 90090.0 * z_d6 - 109395.0 * z_d8 + 46189.0 * z_d10
        ) / d2

    # --- J10 ---
    if nj >= 9 and j_coefficients[8] != 0:
        r4 = r2 * r2
        r6 = r4 * r2
        r8 = r6 * r2
        r10 = r8 * r2
        cr = j_coefficients[8] / r10
        ff = 384 * cr * (
            -0.08203125 + 5.33203125 * z_d2 - 53.3203125 * z_d4
            + 181.2890625 * z_d6 - 246.03515625 * z_d8
            + 114.81640625 * z_d10
        ) / d3
        aj[0] += x * ff
        aj[1] += y * ff
        aj[2] += cr * z * (
            -346.5 + 7507.5 * z_d2 - 45045.0 * z_d4
            + 109395.0 * z_d6 - 115472.5 * z_d8 + 44089.5 * z_d10
        ) / d3

    return aj
