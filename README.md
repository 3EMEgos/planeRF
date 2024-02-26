# planeRF
Python code repository for an interactive web app that calculates and plots $S_E$ and $S_H$ over a height of 2 m of a RF plane wave that is incident at various angles on a PEC or real ground.

The `Dash` directory contains all the code related to the web app, while the main codebase is located within the `src/planeRF` directory.

The features of the planeRF web app include:
+ Two display pages:
  + Introduction page
  + Calculation/plots page
+ A side bar with subheadings for navigating to these pages  
+ Calculate both $S_E=|E|²/377$ and $S_H=377|H|²$
+ Calculate for both transverse magnetic (TM) and transverse electric (TE) modes of the incident plane wave
![TM & TE mode](https://github.com/3EMEgos/planeRF/blob/main/assets/TM-TE-mode.png)
+ Allow user input for:
  + Power density, $S_0$, of the incident plane wave (validated text input box in W/m²)
  + Frequency in MHz (slider input and text box ranging from 1 to 6000 MHz)
  + Spatial averaging length (slider input and text box ranging from 0.1 to 2m in 0.1 increments)
  + Number of spatial averaging points (slider input and text box ranging from 1 to 100 in 1 increments)
  + Selection of [spatial averaging scheme](https://2fc.gitbook.io/spatial-avg-wg/methodology/numerical-approaches-for-spatial-averaging) (combo box) using:
    + Simple averaging
    + Riemman sum
    + Trapezoidal rule
    + Simpsons 1/3 rule
    + Simpsons 3/8 rule
    + Gaussian Legendre quadrature 
  + Selection of PEC or real ground (radio buttons)
  + Dielectric values, $\epsilon_r$ and $\sigma$ (S/m), of real ground (validated text input box)
  + Angle of incidence θ for plane wave (slider input and text box ranging from 0° to 90°)
  + Toggle display (checkbox) of:
    + $S_E$
    + $S_H$
    + $S_0$
    + Spatial average value $S_{sa_E}$ for $S_E$ values
    + Spatial average value $S_{sa_H}$ for $S_H$ values
    + Maximum spatial average value $S_{sa}$ of $S_{sa_E}$ & $S_{sa_H}$
    + Spatial averaging points on plots
    + Weighting for each spatial averaging point (4 decimal places)
    + Single assessment point on plots (with validated text inout box for height of that point, 0 to 2m)
+ Calculated levels of $S_E$ and $S_H$ from z = 0 to 2m are displayed in side-by-side plots for TM and TE modes
+ If selected, $S_0$ $S_{sa}$ $S_{sa_E}$ and $S_{sa_H}$ are displayed as vertical dashed lines
+ The plots will display a legend for the  $S_E$ $S_H$ $S_0$ $S_{sa}$ $S_{sa_E}$ and $S_{sa_H}$ curves
+ The plots will display a grid
+ The vertical plot axis labelled as "z (m)"
+ The horizontal plot axis labelled as "S (W/m²)"
+ A plot/diagram showing the model for the TM or TE mode is shown above each plot, with angle of incidence reflecting input value
+ The $S_{E_{sa}}$ and $S_{H_{sa}}$ values are shown below each TM and TE plot

## Getting started

### Installation

Clone this repository to your local machine:
```bash
git clone git@github.com:3EMEgos/planeRF.git
```
Enter the repository:
```bash
cd planeRF
```
Install `planeRF` preferably within a virtual environment, e.g., by using Conda:
```bash
conda create --name planeRF python=3.10
conda activate planeRF
pip install --upgrade pip
python -m pip install -e .
```
To use the `planeRF` env in Jupyter or Jupyterlab, run the following commands:
```bash
conda activate planeRF
conda install ipykernel
ipython kernel install --user --name=planeRF
```

### Use

Run the web app by simply running the `app.py` file inside the `Dash` directory.

If instead you wish to use the code locally without the web app, you can simply do the following:
```python
from planeRF import compute_power_density


# set up the input values
f = ...     # replace with your desired frequency in Hz
theta = 45  # replace with your desired angle in deg
pol = 'TM'  # define the polarization mode

# perform the computation
SH, SE = compute_power_density(f, theta, pol)
```

### Run the tests

In the root folder of the project, simply run
```bash
pytest tests
```
to run all the unit tests.

## Authors
Vitas Anderson

Yong Cai

Ante Kapetanović

## License
to-be-defined
