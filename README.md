# planeRF
Python code repository for an interactive web app that calculates and plots $S_E$ and $S_H$ over a height of 2m of a RF plane wave that is incident at various angles on a PEC or real ground.

The `Dash##` and `Anvil##` folders contain major iterations of Dash and Anvil web apps.

The features of the planeRF web app include:
+ Two display pages:
  + Introduction page
  + Calculation/plots page
+ A side bar with subheadings for navigating to these pages  
+ Calculate both $S_E=|E|²/377$ and $S_H=377|H|²$
+ Calculate for both transverse magnetic (TM) and transverse electric (TE) modes of the incident plane wave
![TM & TE mode](https://github.com/3EMEgos/planeRF/blob/main/assets/TM_TE_mode.png "TM & TE mode")
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
  + Angle of incidence for plane wave (slider input and text box ranging from 0° to 90°)
  + Toggle display (checkbox) of:
    + $S_E$
    + $S_H$
    + $S_0$
    + Spatial average value, $S_{E_{sa}}$ for $S_E$ values
    + Spatial average value, $S_{H_{sa}}$ for $S_H$ values
    + Spatial averaging points on plots
    + Weighting for each spatial averaging point (4 decimal places)
    + Single assessment point on plots (with validated text inout box for height of that point, 0 to 2m)
+ Calculated levels of $S_E$ and $S_H$ from z = 0 to 2m are displayed in side-by-side plots for TM and TE modes
+ If selected, $S_{sa}$ &/or $S_0$ are displayed as vertical dashed lines
+ The plots will display a legend for the  $S_E$, $S_H$, $S_0$ and $S_{sa}$ curves
+ The plots will display a grid
+ The vertical plot axis labelled as "z (m)"
+ The horizontal plot axis labelled as "S (W/m²)"
+ A plot/diagram showing the model for the TM or TE mode is shown above each plot, with angle of incidence reflecting input value
+ The $S_{E_{sa}}$ and $S_{H_{sa}}$ values are shown below each TM and TE plot 
    
  
