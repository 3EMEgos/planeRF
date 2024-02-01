# planeRF
Python code repository for an interactive web app that calculates and plots $S_E$ and $S_H$ of a RF plane wave that is incident on a PEC or real ground.

The `Dash##` and `Anvil##` folders contain major iterations of Dash and Anvil web apps.

The features of the planeRF web app include:
+ Calculate both $S_E=|E|²/377$ and $S_H=377|H|²$.
+ Calculate for both transverse magnetic (TM) and transverse electric (TE) modes of the incident plane wave.
+ Provide user input for:
  + Power density (W/m²) of the incident plane wave (validated text input box)
  + Frequency in MHz (slider input and text box ranging from 1 to 6000 MHz)
  + Spatial averaging length (slider input and text box ranging from 0.1 to 2m in 0.1 increments)
  + Number of spatial averaging points (slider input and text box ranging from 1 to 100 in 1 increments)
  + Selection of spatial averaging scheme for:
    + Simple
    + Riemman sum
    + Trapezoidal rule
    + Simpsons 1/3 rule
    + Simpsons 3/8 rule
    + Gaussian Legendre quadrature 
  + Selection of PEC or real ground (radio buttons)
  + Dielectric values, $\epsilon_r$ and $\sigma$ (S/m), of real ground (validated text input box)
  + Angle of incidence for plane wave (slider input and text box ranging from 0° to 90°)
  + Toggle display of $S_E$ (checkbox)
  + Toggle display of $S_H$ (checkbox)
  + Toggle display of the spatial average value, $S_{sa}$ (checkbox)
  + Toggle display of power density of the plane wave incident field strength, $S_0$ (checkbox)
  + Toggle display of spatial averaging points (checkbox)
  + Toggle display of single assessment point (checkbox) with height of that point (validated text input, 0 to 2m)
+ Calculated levels of $S_E$ and $S_H$ from z = 0 to 2m are displayed in side-by-side plots for TM and TE modes.
+ Each plot shows a vertical dashed line for $S_{sa}$ &/or $S_0$ if selected
+ A figure showing the model for the TM or TE mode is shown above each plot.
    
  
