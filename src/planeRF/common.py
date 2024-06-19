# Support code for the "Ground Reflection" Dash page
 
import os
import numpy as np
import pandas as pd
from scipy.constants import epsilon_0 as eps0, mu_0 as mu0

__all__ = ["compute_power_density", "sagnd", "compute_ns", "soil_dielectrics",
           "ground_dielectrics"]

# PARAMETERS
SOIL = pd.read_csv(os.path.join("Dash","data","soil.csv"))

def soil_dielectrics(ground_type: str, fMHz: float):
    """
    Calculates the complex dielectric value (ε'-jε") and conductivity (σ) of 
    wet or dry soil (silty loam) at nominated frequency (fMHz) and 
    temperature (T=23°C) indicated in the SOIL dataframe in accordance with ITU-R P.527-6
    INPUTS:
        SOIL: external dataframe containing parameters of wet or dry soil types
        ground_type: type of ground (PEC, Wet Soil or Dry Soil)
        fMHz: exposure frequency in MHz
    INTERMEDIATE VARIABLES:    
        P_Sand, P_Clay, P_Silt, Pb, Ps, mv Parameters from ITU-R P.527-6 Fig.14 and Fig.16
        T = 23 Celsius
        mv = 0.5 for wet soil; mv = 0.07 for dry soil
    OUTPUTS:
        eps__r, eps__i, sigma
    """
    # Extract soil parameters for specified ground type
    soil = SOIL[SOIL.Type == ground_type].values[0]
    soil_type, P_Sand, P_Clay, P_Silt, Ps, Pb, T, mv = soil    

    # Calculate frequency in GHz
    fGHz = fMHz / 1000

    # Calculate dielectric equation parameters
    sigma_1 = (
        0.0467 + 0.2204 * Pb - 0.004111 * P_Sand - 0.006614 * P_Clay
    )  # eq(69)
    sigma_2 = (
        -1.645 + 1.939 * Pb - 0.0225622 * P_Sand + 0.01594 * P_Clay
    )  # eq(70)

    sigma_eff_r = (fGHz / 1.35) * (
        (sigma_1 - sigma_2) / (1 + (fGHz / 1.35) ** 2)
    )  # eq(67)
    sigma_eff_i = sigma_2 + (
        (sigma_1 - sigma_2) / (1 + (fGHz / 1.35) ** 2)
    )  # eq(68)

    Θ = 300 / (T + 273.15) - 1  # eq(11)
    eps_s = 77.66 + 103.3 * Θ  # eq(8)
    eps_1 = 0.0671 * eps_s  # eq(9)
    eps_inf = 3.52 - 7.52 * Θ  # eq(10)

    f1 = 20.20 - 146.4 * Θ + 316 * Θ**2  # eq(12)
    f2 = 39.8 * f1  # eq(13)

    # Calculate soil dielectric values
    # eps_fw_r and eps_fw_i are the real and the imaginary parts 
    # of the complex relative permittivity of free water
    eps_fw_r = (
        (eps_s - eps_1) / (1 + (fGHz / f1) ** 2)
        + (eps_1 - eps_inf) / (1 + (fGHz / f2) ** 2)
        + eps_inf
        + (18 * sigma_eff_r / fGHz) * ((Ps - Pb) / (Ps * mv))
    )  # eq(65)

    eps_fw_i = (
        (fGHz / f1) * (eps_s - eps_1) / (1 + (fGHz / f1) ** 2)
        + (fGHz / f2) * (eps_1 - eps_inf) / (1 + (fGHz / f2) ** 2)
        + (18 * sigma_eff_i / fGHz) * ((Ps - Pb) / (Ps * mv))
    )  # eq(66)

    eps_sm_r = (1.01 + 0.44 * Ps) ** 2 - 0.062  # eq(61)
    β_r = 1.2748 - 0.00519 * P_Sand - 0.00152 * P_Clay  # eq(62)
    β_i = 1.33797 - 0.00603 * P_Sand - 0.00166 * P_Clay  # eq(63)
    α = 0.65  # eq(64)

    eps__r = (
        1 + (Pb / Ps) * (eps_sm_r**α - 1) + (mv**β_r) * (eps_fw_r**α) - mv
    ) ** (1 / α)
    eps__i = ((mv**β_i) * (eps_fw_i**α)) ** (1 / α)
    sigma = 0.05563 * fGHz * eps__i  # eq(3a)

    return eps__r, eps__i, sigma


def ground_dielectrics(ground_type: str, fMHz: float):
    '''
    Calculate the dielectric properties of the ground
    INPUTS:
        ground_type = type of ground (PEC, Wet Soil or Dry Soil)
        fMHz = exposure frequency in MHz
    OUTPUTS:
        epsr = real part of complex permittivity
        sigma = conductivity
    '''

    match ground_type:
        case 'PEC Ground':  # PEC Ground, lossless
            epsr = [1, 10]
            sigma = [0, 1e6]
        case "Wet Soil" | "Dry Soil":  # Wet or Dry Soil
            er, ei, sigma_i = soil_dielectrics(ground_type, fMHz)
            epsr = [1, er]
            sigma = [0, sigma_i]
        case _:
            raise ValueError(f'unknown ground_type ({ground_type} has been specified)')
        
    return epsr, sigma

            
def compute_ns(freqMHz: float, L: float):
    """
    INPUTS:
        freqMHz = Frequency in MHz
        L = Height of Human Body in meters
    OUTPUTS:
        ns = number of sampling points/plotting points. Make ns as even number
        Assume at least 8 points to plot a reasonable smooth curve.
        The standing wave has twice the frequency than the frequency of interest.
    """

    Ns = max(200, int(np.round((8 * L * 2) / (300 / freqMHz))))
    if Ns % 2 == 1:
        Ns = Ns + 1
    return Ns


def sagnd(kind: str, n: int, L: float):
    """Calculate height (z) and weighting (w) of spatial averaging points
    for various schemes over ground
    INPUTS:
      kind = spatial averaging scheme ('ps','simple', 'RS', 'S13', 'S38', 'GQR')
      n = number of spatial averaging points
      L = spatial averaging length, or height of point for 'ps' case
    OUTPUTS:
      z = np array of spatial avg assessment point heights (z=0 at ground level)
      w = np array of assessment point weights
    """

    # Assertion tests on input data
    kinds = ("ps", "RS", "Simple", "S13", "S38", "GQR")
    assert kind in kinds, f"kind {kind} must be one of {kinds}"
    assert 0 < L <= 2, f"L ({L}) must be >0 and ≤2"
    assert type(n) == int
    assert 1 <= n <= 640, f"n ({n}) must be ≥1 and ≤640"

    # Select case for spatial averaging scheme
    match kind:
        case "ps":
            # point spatial
            z = np.array([float(L)])
            w = np.array([1.0])

        case "Simple":
            # Simple average
            assert n >= 2, f"n ({n}) must be >= 2"
            z = np.linspace(0, L, n + 1)
            w = np.ones(n + 1) / (n + 1)

        case "RS":
            # Riemann Sum
            assert n >= 2, f"n ({n}) must be >= 2"
            z = np.linspace(L / (2 * n), L - L / (2 * n), n)
            w = np.ones(n) / n

        case "GQR":
            # Gaussian Legendre Quadrature Rule
            z, w = np.polynomial.legendre.leggauss(n)
            z = (z + 1) * L / 2
            w = w / 2

        case "S13":
            # Simpsons 1/3 rule
            assert (
                n >= 2
            ), f"n ({n}) must be >= 2"  # n denotes the number of intervals
            assert (
                n + 1
            ) % 2 == 1, (  # make sure the number of intervals is even number
                f"n ({n}) must be even number"
            )
            z = np.linspace(0, L, n + 1)
            w = np.ones(n + 1)  # initialize the weights with n+1 numbers of 1
            wts = [4.0, 2.0] * int(
                n / 2
            )  # create the middle part of the weights wts
            wts.pop()  # drop the last weight of wts
            w[1:n] = wts[0 : n - 1]  # replace the middle part of the weights
            # for i in range(n-1):
            #     w[i+1] = wts[i]
            w = w * L / (3 * n)

        case "S38":
            # Simpsons 3/8 Rule
            assert n >= 4, f"n ({n}) must be >= 4"
            assert (
                n - 1
            ) % 3 == 0, f"number of intervals ({n-1}) must be divisible by 3"
            z = np.linspace(0, L, n)
            w = np.ones(n)
            wts = [3.0, 3.0, 2.0] * 300  # a pop list for the weights
            for i in range(n - 2):
                w[i + 1] = wts.pop(0)
            w = w / sum(w)

    return z, w


def compute_power_density(ground_type: str, S0: float, fMHz: float, 
                          theta: float, pol: str, z):
    """Calculate the plane wave equivalent power density levels of E & H above ground
    for a plane wave in air that is obliquely incident on a PEC or real ground

    Parameters
    ----------
    ground_type : {'PEC Ground', 'Wet Soil', 'Dry Soil'}
        Type of ground
    S0 : float
        Power flux density of the incident plane wave in W/m²
    freqMHz : float
        Frequency of the plane wave in MHz.
    theta : float
        Angle of incidence in degrees of the plane wave, i.e. incoming angle of the 
        propagation vector (k) relative to the normal of the first interface.
    pol : {'TM', 'TE'}
        Polarization of the plane wave.
        'TM' = Transverse Magnetic polarization (H parallel to interface)
        'TE' = Transverse Electric polarization (E parallel to interface)
    z : numpy float array
        height of points above ground in m. Note that ground level is 0 and
        increasing heights above ground are -ve

    Returns
    -------
    tuple : (SH, SE)
        Equivalent power density levels for H and E calculated at z heights

    Notes
    -----
    The algorithm is based on the theory found in Weng Cho Chew,
    'Waves and Fields in Inhomogeneous Media,' IEEE PRESS Series on
    Electromagnetic Waves. The code was adapted from a Matlab script
    developed by Kimmo Karkkainen.
    """
    # Data input checks
    POLS = ('TE','TM')
    assert pol in POLS, f'pol ({pol}) must be in {POLS}'
    assert 0 <= theta <= 90, f'theta ({theta}) must be within range of 0° to 90°'

    # initialize settings
    N = 2  # number of layers
    Z0 = np.sqrt(mu0 / eps0)  # free-space impedance
    zi = [0]  # interface level between layer 1 (air) and 2 (ground)
    zi.append(1e9)  # add a very large z value to act as infinity
    mur = [1, 1]  # relative permeability of layers 1 and 2
    w = 2.0 * np.pi * fMHz * 1e6  # angular frequency
    theta = np.deg2rad(theta)  # convert theta from degrees to radians
    E0 = np.sqrt(2 * S0 * Z0)  # calculate peak E-field level of S0

    # Set permittivity and conductivity of ground
    epsr, sigma = ground_dielectrics(ground_type, fMHz)

    # wavenumber
    eps = [er * eps0 + s / (1j * w) for er, s in zip(epsr, sigma)]
    mu = [ur * mu0 for ur in mur]
    K = [w * np.sqrt(e * mu0) for e in eps]
    Kz = [0] * N
    Kz[0] = K[0] * np.cos(theta)
    Kt = K[0] * np.sin(theta)
    for L in range(1, N):
        SQ = K[L] ** 2 - Kt**2
        SQr = np.real(SQ)
        SQi = np.imag(SQ)
        if (SQi == 0) & (SQr < 0):
            Kz[L] = -np.sqrt(SQ)
        else:
            Kz[L] = np.sqrt(SQ)

    # reflection and transmission coefficients for interfaces
    R = np.empty((N, N), dtype=complex)
    T = R.copy()
    for k in range(N - 1):
        if pol == "TM":  # reflection coefficient for the H field
            e1, e2 = eps[k + 1] * Kz[k], eps[k] * Kz[k + 1]
            R[k, k + 1] = (e1 - e2) / (e1 + e2)
        elif pol == "TE":  # reflection coefficient for the E field
            m1, m2 = mu[k + 1] * Kz[k], mu[k] * Kz[k + 1]
            R[k, k + 1] = (m1 - m2) / (m1 + m2)
        else:
            raise Exception(f"pol ({pol}) must be either TE or TM")
        R[k + 1, k] = -R[k, k + 1]
        T[k, k + 1] = 1 + R[k, k + 1]
        T[k + 1, k] = 1 + R[k + 1, k]

    # generalized reflection coefficients for interfaces
    gR = np.zeros(N, dtype=complex)
    for k in range(N - 1)[::-1]:
        thickness = zi[k + 1] - zi[k]  # layer thickness
        g = gR[k + 1] * np.exp(-2 * 1j * Kz[k + 1] * thickness)
        gR[k] = (R[k, k + 1] + g) / (1 + R[k, k + 1] * g)

    # amplitudes
    A = np.zeros(N, dtype=complex)
    if pol == "TM":  # amplitude of the H field in layer 1
        A[0] = np.abs(K[0] / (w * mu[0]))
    else:  # amplitude of the E field in layer 1
        A[0] = 1
    for k in range(1, N):  # amplitudes in other layers
        A[k] = (
            A[k - 1]
            * np.exp(-1j * Kz[k - 1] * zi[k - 1])
            * T[k - 1, k]
            / (
                1
                - R[k, k - 1]
                * gR[k]
                * np.exp(-2 * 1j * Kz[k] * (zi[k] - zi[k - 1]))
            )
            / np.exp(-1j * Kz[k] * zi[k - 1])
        )

    # form a vector that tells in which layer the calculation points are located
    zl = []
    layer = 0
    for zp in z:
        while zp >= zi[layer]:
            layer += 1
        zl.append(layer)

    # E-field and H-field components
    Azl = np.array([A[z] for z in zl])
    Kzzl = np.array([Kz[z] for z in zl])
    Kzl = np.array([K[z] for z in zl])
    gRzl = np.array([gR[z] for z in zl])
    epszl = np.array([eps[z] for z in zl])
    zizl = np.array([zi[z] for z in zl])
    sigmazl = np.array([sigma[z] for z in zl])

    if pol == "TM":  # the forward and backward Hf and Hb have only y component
        Hf = Azl * np.exp(-1j * Kzzl * z)
        Hb = Azl * gRzl * np.exp(-2 * 1j * Kzzl * zizl + 1j * Kzzl * z)
        Ex = Z0 * np.cos(theta) * (Hf - Hb)
        Ez = (-Z0) * np.sin(theta) * (Hf + Hb)
        SH = 0.5 * E0**2 * Z0 * (np.abs(Hf + Hb)) ** 2
        SE = 0.5 * E0**2 * (1 / Z0) * (np.abs(Ex) ** 2 + np.abs(Ez) ** 2)
    else:  # the forward and backward Ef and Eb have only y component
        Ef = Azl * np.exp(-1j * Kzzl * z)
        Eb = Azl * gRzl * np.exp(-2 * 1j * Kzzl * zizl + 1j * Kzzl * z)
        Hx = -(1 / Z0) * np.cos(theta) * (Ef - Eb)
        Hz = (1 / Z0) * np.sin(theta) * (Ef + Eb)
        SH = 0.5 * E0**2 * Z0 * (np.abs(Hx) ** 2 + np.abs(Hz) ** 2)
        SE = 0.5 * E0**2 * (1 / Z0) * (np.abs(Ef + Eb)) ** 2

    return SH, SE