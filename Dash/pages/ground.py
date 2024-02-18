import dash
from dash import dcc, callback, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objs as go

from planeRF import compute_power_density


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
        result = compute_power_density(frequency, angle, 'TM')
        if result is not None:
            trace1 = go.Scatter(
                y=np.linspace(-2, 0, 2001),
                x=result[0],
                mode='lines',
                name='SH'
            )
            trace2 = go.Scatter(
                y=np.linspace(-2, 0, 2001),
                x=result[1],
                mode='lines',
                name='SE'            
            )
            layout = go.Layout(
                title=f'TM mode, {frequency:g} MHz, theta = {angle:g}°',
                xaxis=dict(title='Power Flux Density (W/m2)'),
                yaxis=dict(title='z (m)'),
            )
            
            # fig = go.Figure(data=[trace], layout=layout)
            fig = go.Figure(layout=layout)
            fig.add_trace(trace1)
            fig.add_trace(trace2)
            return fig
    elif angle is None or frequency is None:
        raise dash.exceptions.PreventUpdate


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
        result = compute_power_density(frequency, angle, 'TE')
        if result is not None:
            trace1 = go.Scatter(
                y=np.linspace(-2, 0, 2001),
                x=result[0],
                mode='lines',
                name='SH'
            )
            trace2 = go.Scatter(
                y=np.linspace(-2, 0, 2001),
                x=result[1],
                mode='lines',
                name='SE'
            )
            layout = go.Layout(
                title=f'TE mode, {frequency:g} MHz, theta = {angle:g}°',
                xaxis=dict(title='Power Flux Density (W/m2)'),
                yaxis=dict(title='z (m)'),
            )
            # fig = go.Figure(data=[trace], layout=layout)
            fig = go.Figure(layout=layout)
            fig.add_trace(trace1)
            fig.add_trace(trace2)
            return fig
    elif angle is None or frequency is None:
        raise dash.exceptions.PreventUpdate


dash.register_page(
    __name__,
    path='/ground',
    title='Ground Plane Reflection',
    name='Ground Plane Reflection'
)

dash.page_container.style = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "1rem 1rem",
}

layout_ground = html.Div([
    # html.Div(children='Plane Wave Incident onto Ground',
    #          style={'textAlign': 'center', 'color': 'blue', 'fontSize': 20}),
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
                      max = 90.0,  # limit the angle range from 0 to 90 deg
                      style={'width':'30%'}),
            html.Br(),
            html.Br(),
            html.H6('Frequency (MHz)',
                    style={'display': 'inline-block',
                    'margin-right': 43}),
            dcc.Input(id='frequency-input', 
                      type='number', 
                      placeholder='', 
                      min = 100, 
                      max = 60000,  # limit the freq range from 0.1 to 60 GHz
                      style={'width':'30%'}),
            html.Br(),
            html.Br(),
            dbc.Button('Run', id='run-button', style={'width':'30%'}, n_clicks=0)
            ],
            width = {'size' : 4}
        ),
        dbc.Col([
            html.H5('TM mode',style={'textAlign':'center'}),
            dcc.Graph(id ='output-graphTM',
                      figure={},
                      style = {'box-shadow': '6px 6px 6px lightgrey'}),
            ],
            width = {'size' : 4}
        ),
        dbc.Col([
            html.H5('TE mode',style={'textAlign':'center'}),
            dcc.Graph(id ='output-graphTE', figure={}, 
                        style = {'box-shadow': '6px 6px 6px lightgrey'})
            ],
            width = {'size' : 4}
        )
    ])
])

layout = dbc.Container([
    html.H4('Plane Wave Incident onto PEC Ground Plane',
    style={'textAlign':'center', 'color':'blue'}),
    html.Hr(),
    layout_ground
])
