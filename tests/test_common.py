import numpy as np
import pytest
import pandas as pd
from planeRF import compute_S_params, compute_power_density

def test_compute_power_density_output_type():
    '''Check compute_power_density output types'''

    # test input values
    fMHz = 1000  # replace with your desired frequency
    theta = 30   # replace with your desired angle
    pol = "TM"
    S0 = 100
    z = np.linspace(-1, 1)

    # perform the computation
    S_params, (epsr, sigma) = compute_S_params("PEC Ground",fMHz,theta,pol)
    SH, SE = compute_power_density(z, S0, *S_params)
    # SH, SE = compute_power_density("PEC Ground", 100, fMHz, theta, pol, z)

    # check the output types
    assert isinstance(SH, np.ndarray)
    assert isinstance(SE, np.ndarray)


def test_compute_power_density_output_values():
    '''Check compute_power_density output values for a wide range of inputs'''

    # Read in independent comparison values from FEKO analysis
    dfFEKO = pd.read_csv('./tests/FEKO_S_values.csv')

    # Set S0 value that was used for FEKO analysis
    S0 = 13.272093639924964
 
    # Assertion tests for testing compute_power_density function values against FEKO data
    for ix, r in dfFEKO.iterrows():
        S_FEKO = r.S
        S_params, (epsr, sigma) = compute_S_params(r.gnd,r.fMHz,r.theta,r.pol)
        SH, SE = compute_power_density([-r.z], S0, *S_params)
        # SH, SE = compute_power_density(r.gnd,S0,r.fMHz,r.theta,r.pol,[-r.z])
        SH_planeRF = SH[0]
        SE_planeRF = SE[0]
        
        S_type = r['S type']
        rtol = 0.001  # relative test tolerance between FEKO and planeRF values
        err_txt = f'\nS values do not match within {rtol*100}% tolerance for '
        
        if S_type == 'SE':
            assert np.isclose(SE_planeRF, S_FEKO, rtol), \
                err_txt + f'{r.gnd}, {r.pol}, {r.fMHz}MHz, {r.theta}°, {r.z}m\n  planeRF SE = {SE_planeRF}\n     FEKO SE = {S_FEKO}\n relative error = {round(100*(SE_planeRF-S_FEKO)/S_FEKO,4)}%'
        elif S_type == 'SH':
            assert np.isclose(SH_planeRF, S_FEKO, rtol), \
                err_txt + f'{r.gnd}, {r.pol}, {r.fMHz}MHz, {r.theta}°, {r.z}m\n  planeRF SH = {SH_planeRF}\n     FEKO SH = {S_FEKO}\n relative error = {round(100*(SH_planeRF-S_FEKO)/S_FEKO,4)}%'

if __name__ == "__main__":
    pytest.main()
