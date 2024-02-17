import dash
import dash_bootstrap_components as dbc
from dash import dcc, callback, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import numpy as np

dash.register_page(__name__,
    path='/ground',
    title='Ground Plane Reflection',
    name='Ground Plane Reflection'
)

dash.page_container.style = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "1rem 1rem",
}

Layout_Ground = html.Div([
            # html.Div(children='Plane Wave Incident onto Ground',
            #                         style={'textAlign': 'center', 'color': 'blue', 'fontSize': 20}),
            dbc.Row([                
                dbc.Col([                    
                        html.H5('Input Parameters', 
                                style={'textAlign':'center'}),                
                        html.H6(['Angle of Incidence (', html.Sup('o'),')'], 
                                style={'display':'inline-block',
                                'margin-right':25}),
                        dcc.Input(id='angle-input', 
                                    type='number', 
                                    placeholder='',
                                    min = 0.0,
                                    max = 90.0, #limit the angle range from 0 to 90degs
                                    style={'width':'30%'}),
                        html.Br(),
                        html.Br(),
                        html.H6('Frequency (MHz)',
                                style={'display':'inline-block',
                                'margin-right':43
                                }),
                        dcc.Input(id='frequency-input', 
                                    type='number', 
                                    placeholder='', 
                                    min = 100, 
                                    max = 60000, #limit the freq range from 100MHz to 60GHz
                                    style={'width':'30%'}),
                        html.Br(),
                        html.Br(),
                        dbc.Button('Run', id='run-button', style={'width':'30%'}, n_clicks=0)
                        ], 
                        width = {'size' : 4}),
            
                dbc.Col([
                    html.H5('TM mode',style={'textAlign':'center'}),
                    dcc.Graph(id ='output-graphTM', figure={},
                              style = {'box-shadow': '6px 6px 6px lightgrey'}),
                        ],
                        width = {'size' : 4}),
                dbc.Col([
                    html.H5('TE mode',style={'textAlign':'center'}),
                    dcc.Graph(id ='output-graphTE', figure={}, 
                              style = {'box-shadow': '6px 6px 6px lightgrey'})
                        ],
                        width = {'size' : 4}
                )
            ])
        ])

layout = dbc.Container(
                [html.H4('Plane Wave Incident onto PEC Ground Plane',
                         style={'textAlign':'center', 'color':'blue'}),
                html.Hr(),
                Layout_Ground])

def Ground(f, θ, pol):
    '''
    Calculates power density (SE=0.5|E|²/Z0, SH=0.5*377|H|²) for a
    plane wave travelling through a multilayered infinite planar medium.
    The first and last layers have infinite thickness.
    Theory on which the algorithm is based can be found e.g. in Weng
    Cho Chew, "Waves and Fields in Inhomogeneous Media", IEEE PRESS
    Series on Electromagnetic Waves. This code was adapted from
    a Matlab script developed by Kimmo Karkkainen.

    INPUTS:
    E0 = E-field level of incident plane wave (V/m peak)
     z = list of z coords where |E|, SAR field levels are evaluated (m)
    zi = list of z coords for interfaces between planar layers (m)
    εr = list of relative permittivity for each layer
     σ = list of conductivity (S/m) for each layer
     f = frequency (Hz) of the planewave
     θ = incoming angle of the propagation vector (k)
         relative to the normal of the first interface. 
    pol = polarization of the plane wave:
          'TM' = TM polarized wave (H parallel to interface)
          'TE' = TE polarized wave (E parallel to interface)
    OUTPUTS:       
    SE = Equivalent plane wave power density (SE = 0.5|E|²/Z0) at specified points (z)
    SH = Equivalent plane wave power density (SH = 0.5|H|²*Z0) at specified points (z)
    '''

    # Set constants
    Z0 = 376.730313668           # free space impedance
    ε0 = 8.854187817620389e-12   # permittivity of free space
    µ0 = 1.2566370614359173e-06  # permeability of free space
    π = np.pi

    E0 = 100                    # E-field of incident plane (V/m peak)
    z = np.linspace(-2,0,2001)  # z-direction coordinates
    zi = [0]                    # z coord of interface between layer 1 and 2
    εr = [1, 10]                # relative permittivities of layers 1 and 2
    μr = [1, 1]                 # relative permeability of layers 1 and 2
    σ = [0, 1e6]              # Assume lossless
    
    # Initialise variables    
    z = np.array(z).round(8)
    θ = θ * π / 180.   # convert θ from degrees to radians
    zi.append(1e9)     # add a very large z coord for 'infinite' last layer
    N = len(εr)        # N is the no. of layers
    w = 2. * π * f * 1e6     # angular frequency 
    i = 1j             # i is imaginary unit
    
    # Wavenumber and its z-component in layers
    ε = [er*ε0 + s/(i*w) for er, s in zip(εr,σ)]
    μ = [ur*µ0 for ur in μr]
    K = [w*np.sqrt(e*µ0) for e in ε]
    Kz = [0]*N
    Kz[0] = K[0] * np.cos(θ)
    Kt    = K[0] * np.sin(θ)

    for L in range(1, N):
        SQ = K[L]**2 - Kt**2
        SQr = np.real(SQ)
        SQi = np.imag(SQ)
        if (SQi == 0) & (SQr < 0):
            Kz[L] = -np.sqrt(SQ)
        else:
            Kz[L] = np.sqrt(SQ)

    # Calculate reflection and transmission coefficients for interfaces
    R = np.zeros((N,N), dtype=complex)
    T = R.copy()
    for k in range(N-1):
        if pol == 'TM':
            # Reflection coefficient for magnetic field
            e1, e2 = ε[k+1] * Kz[k], ε[k] * Kz[k+1]
            R[k,k+1] = (e1 - e2) / (e1 + e2)
        elif pol == 'TE':
            # Reflection coefficient for electric field
            m1, m2 = μ[k+1] * Kz[k],  μ[k] * Kz[k+1]
            R[k,k+1] = (m1 - m2) / (m1 + m2)
        else:
            raise Exception(f'pol ({pol}) must be either TE or TM')

        R[k+1,k] = -R[k,k+1] 
        T[k,k+1] = 1 + R[k,k+1]
        T[k+1,k] = 1 + R[k+1,k]

    # Calculate generalized reflection coefficients for interfaces:
    gR = np.zeros(N, dtype=complex)
    for k in range(N-1)[::-1]:
        thickness = zi[k+1] - zi[k]  # layer thickness
        g = gR[k+1] * np.exp(-2*i*Kz[k+1] * thickness)
        gR[k] = (R[k,k+1] + g) / (1 + R[k,k+1] * g)

    A = np.zeros(N, dtype=complex)
    if pol == 'TM':
        A[0] = np.abs(K[0] / (w*μ[0]))  # Amplitude of magnetic field in layer 1
    else:
        A[0] = 1                        # Amplitude of electric field in layer 1

    # Calculate amplitudes in other layers:
    for k in range(1, N):
        A[k] = A[k-1] * np.exp(-i*Kz[k-1]*zi[k-1])*T[k-1,k] \
            /(1-R[k,k-1] * gR[k] * np.exp(-2*i*Kz[k]*(zi[k]-zi[k-1]))) \
            /np.exp(-i*Kz[k]*zi[k-1])

    # Form a vector that tells in which layer the calculation points are located:
    zl = []
    layer = 0
    for zp in z:
        while zp >= zi[layer]:
            layer += 1
        zl.append(layer)

    # Calculate E-field:
    Azl = np.array([A[z] for z in zl])
    Kzzl = np.array([Kz[z] for z in zl])
    Kzl = np.array([K[z] for z in zl])
    gRzl = np.array([gR[z] for z in zl])
    εzl = np.array([ε[z] for z in zl])
    zizl = np.array([zi[z] for z in zl])
    σzl = np.array([σ[z] for z in zl])


    if pol == 'TM':
        # The forward and backward Hf and Hb have only y component

        Hf = Azl * np.exp(-i*Kzzl*z)
        Hb = Azl * gRzl * np.exp(-2*i*Kzzl*zizl + i*Kzzl*z)
        Ex = Z0 * np.cos(θ) * (Hf-Hb)
        Ez = (-Z0) * np.sin(θ) * (Hf+Hb)
        SH = 0.5 * E0 ** 2 * Z0 * (np.abs(Hf+Hb))**2
        SE = 0.5 * E0 ** 2 * (1/Z0) * (np.abs(Ex)**2 + np.abs(Ez)**2)

    else:
        # The forward and backward Ef and Eb have only y component
        Ef = Azl * np.exp(-i*Kzzl*z)
        Eb = Azl * gRzl * np.exp(-2*i*Kzzl*zizl + i*Kzzl*z)
        Hx = -(1/Z0) * np.cos(θ) * (Ef-Eb)
        Hz = (1/Z0) * np.sin(θ) * (Ef+Eb) 
        SH = 0.5 * E0 ** 2 * Z0 * (np.abs(Hx)**2 + np.abs(Hz)**2)
        SE = 0.5 * E0 ** 2 * (1/Z0) * (np.abs(Ef+Eb))**2

    return SH, SE

# Update TM graph
@callback(
    Output(component_id='output-graphTM', component_property='figure'),
    [Input('run-button', 'n_clicks')],
    [State('angle-input', 'value'),
    State('frequency-input', 'value')]
)
def update_graph_TM(n_clicks, angle, frequency):

    if n_clicks > 0 and angle is not None and frequency is not None:
        angle = float(angle)
        frequency = float(frequency)
        # print(angle)
        # print(frequency)
        result = Ground(frequency,angle,'TM')

        if result is not None:
            trace1 = go.Scatter(
                y = np.linspace(-2,0,2001),
                x = result[0],
                mode='lines',
                name='SH'                
            )

            trace2 = go.Scatter(
                y = np.linspace(-2,0,2001),
                x = result[1],
                mode='lines',
                name='SE'            
            )
            
            layout = go.Layout(
                title=f'TM mode, {frequency:g} MHz, θ = {angle:g}°',
                xaxis=dict(title='Power Flux Density (W/m²)'),
                yaxis=dict(title='z (m)'),
                # width=300,
                # height=500
            )
            
            # fig = go.Figure(data=[trace], layout=layout)
            fig = go.Figure(layout=layout)
            fig.add_trace(trace1)
            fig.add_trace(trace2)

            return fig
        
    elif angle is None or frequency is None:
        raise dash.exceptions.PreventUpdate

# Update TE graph
@callback(
    Output(component_id='output-graphTE', component_property='figure'),
    [Input('run-button', 'n_clicks')],
    [State('angle-input', 'value'),
    State('frequency-input', 'value')]
)
def update_graph_TE(n_clicks, angle, frequency):

    if n_clicks > 0 and angle is not None and frequency is not None:
        angle = float(angle)
        frequency = float(frequency)

        # print(angle)
        # print(frequency)
        result = Ground(frequency,angle,'TE')

        if result is not None:
            trace1 = go.Scatter(
                y = np.linspace(-2,0,2001),
                x = result[0],
                mode='lines',
                name='SH'                
            )

            trace2 = go.Scatter(
                y = np.linspace(-2,0,2001),
                x = result[1],
                mode='lines',
                name='SE'            
            )
            
            layout = go.Layout(
                title=f'TE mode, {frequency:g} MHz, θ = {angle:g}°',
                xaxis=dict(title='Power Flux Density (W/m²)'),
                yaxis=dict(title='z (m)'),
                # width=300,
                # height=500
            )
            
            # fig = go.Figure(data=[trace], layout=layout)
            fig = go.Figure(layout=layout)
            fig.add_trace(trace1)
            fig.add_trace(trace2)

            return fig
        
    elif angle is None or frequency is None:
        raise dash.exceptions.PreventUpdate
