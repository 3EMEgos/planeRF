import numpy as np
from scipy.constants import epsilon_0 as eps0, mu_0 as mu0


__all__ = ["compute_power_density", "sagnd", "compute_ns"]

def compute_ns(freqMHz:float, L:float):
    '''
    INPUTS:
        freqMHz = Frequency in MHz
        L = Height of Human Body in meters
    OUTPUTS:
        ns = number of sampling points/plotting points. Make ns as even number
        Assume at least 8 points to plot a reasonable smooth curve (sinusoidal curve).
        The standing wave has twice the frequency than the frequency of interest.
    '''

    # assume at least 8 points to plot a reasonable smooth curve (sinusoidal curve).
    Ns = max(200, int(np.round((8*L*2)/(300/freqMHz))))
    if (Ns % 2 == 1):
        Ns = Ns + 1
    return Ns

def sagnd(kind:str, n:int, L:float):
    '''Calculate height (z) and weighting (w) of spatial averaging points
       for various schemes over ground
       INPUTS:
         kind = spatial averaging scheme ('ps','simple', 'RS', 'S13', 'S38', 'GQR')
         n = number of spatial averaging points
         L = spatial averaging length, or height of point for 'ps' case
       OUTPUTS:
         z = np array of spatial avg assessment point heights (z=0 at ground level)
         w = np array of assessment point weights
    '''

    # Assertion tests on input data
    kinds = ('ps','RS','Simple','S13','S38','GQR')
    assert kind in kinds, f'kind {kind} must be one of {kinds}'
    assert 0 < L <= 2, f'L ({L}) must be >0 and ≤2'
    assert type(n) == int
    assert 1 <= n <= 640, f'n ({n}) must be ≥1 and ≤640'

    # Select case for spatial averaging scheme
    match kind:
        case 'ps':
            # point spatial
            z = np.array([float(L)])
            w = np.array([1.])

        case 'Simple':
            # Simple average
            assert (n >= 2), f"n ({n}) must be >= 2"
            z = np.linspace(0, L, n+1)
            w = np.ones(n+1)/(n+1)

        case 'RS':
            # Riemann Sum
            assert (n >= 2), f"n ({n}) must be >= 2"
            z = np.linspace(L/(2*n), L-L/(2*n), n)
            w = np.ones(n) / n

        case 'GQR':
            # Gaussian Legendre Quadrature Rule
            z, w = np.polynomial.legendre.leggauss(n)
            z = (z + 1) * L/2
            w = w / 2

        case 'S13':
            # Simpsons 1/3 rule
            assert (n >= 2), f"n ({n}) must be >= 2" # n denotes the number of intervals
            assert ((n+1)%2 == 1), f"n ({n}) must be even number" # make sure the number of intervals is even number
            z = np.linspace(0,L,n+1)
            w = np.ones(n+1) # initialize the weights with n+1 numbers of 1
            wts = [4.,2.]*int(n/2)  # create the middle part of the weights wts
            wts.pop() # drop the last weight of wts
            w[1:n] = wts[0:n-1] # replace the middle part of the weights
            # for i in range(n-1):
            #     w[i+1] = wts[i]
            w = w * L/(3*n) 

        case 'S38':
            # Simpsons 3/8 Rule
            assert (n >= 4), f"n ({n}) must be >= 4"
            assert ((n-1)%3 == 0), f"number of intervals ({n-1}) must be divisible by 3"
            z = np.linspace(0,L,n)
            w = np.ones(n)
            wts = [3.,3.,2.]*300  # a pop list for the weights
            for i in range(n-2):
                w[i+1] = wts.pop(0)
            w = w / sum(w)

    return z, w

def compute_power_density(ground_type, S0, f, theta, pol, z):
    """Returns power density for a plane wave traveling through a
    multilayered infinite planar medium where the first and last layers
    have infinite thickness.

    Parameters
    ----------
    ground_type : string
        Type of ground, either 'PEC Ground' or any other string for
        real ground.
    E0 : float
        Electric field magnitude of the incident plane wave (V/m).
    f : float
        Frequency (Hz) of the plane wave.
    theta : float
        Incoming angle of the propagation vector (k) relative to the
        normal of the first interface.
    pol : {'TM', 'TE'}
        Polarization of the plane wave.
        'TM' = TM polarized wave (H parallel to interface)
        'TE' = TE polarized wave (E parallel to interface)

    Returns
    -------
    tuple
        Containing the equivalent plane wave power density (SH, SE, S0)
    
    Notes
    -----
    The algorithm is based on the theory found in Weng Cho Chew,
    'Waves and Fields in Inhomogeneous Media,' IEEE PRESS Series on
    Electromagnetic Waves. The code was adapted from a Matlab script
    developed by Kimmo Karkkainen.
    """
    # initialize settings
    Z0 = np.sqrt(mu0 / eps0)  # free-space impedance
    
    zi = [0]  # interface level between layer 1 and 2
    epsr = [1, 1]  # relative permittivities of layers 1 and 2
    mur = [1, 1]  # relative permeability of layers 1 and 2
    # sigma = [0, 1e6]  # lossless conditions
    zi.append(1e9)  # add a very large z value to act as infinity
    N = len(epsr)  # number of layers
    w = 2.0 * np.pi * f * 1e6  # angular frequency
    theta = np.deg2rad(theta)

    E0 = np.sqrt(2*S0*Z0)

    if ground_type == 'PEC Ground':
        epsr = [1, 10]
        sigma = [0, 1e6]  # PEC Ground, lossless
    else:
        epsr = [1, 10]
        sigma = [0, 0.1]  # Real Ground

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
            / (1 - R[k, k - 1] * gR[k]
               * np.exp(-2 * 1j * Kz[k] * (zi[k] - zi[k - 1])))
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
