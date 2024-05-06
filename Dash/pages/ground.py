import dash
from dash import dcc, callback, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objs as go

from planeRF import compute_power_density, sagnd, compute_ns


@callback(
    Output(component_id="output_graph_TM", component_property="figure"),
    [
        Input("run_button", "n_clicks"),
        Input("ground_radioitem", "value"),
        Input("sa_method_dpdn", "value"),
    ],
    [
        State("S0_input", "value"),
        State("angle_input", "value"),
        State("fMHz_input", "value"),
    ],
)
def update_graph_TM(
    n_clicks, ground_type, sa_method_dpdn, S0, angle, frequency
):
    if (
        n_clicks > 0
        and S0 is not None
        and angle is not None
        and frequency is not None
    ):
        L = 2.0
        ground_type = str(ground_type)
        sa_method_dpdn = str(sa_method_dpdn)
        S0 = float(S0)
        angle = float(angle)
        frequency = float(frequency)
        Ns = compute_ns(frequency, L)
        Nt = Ns + 1
        z = np.linspace(-2, 0, Nt)  # z-direction coordinates
        z[-1] = -1E-6 # make the last z value slightly below zero
        result = compute_power_density(
            ground_type, S0, frequency, angle, "TM", z
        )
        Zs, Ws = sagnd(sa_method_dpdn, Ns, L)
        Ssa_h = np.dot(result[0][0:Nt:1], Ws) / L
        Ssa_e = np.dot(result[1][0:Nt:1], Ws) / L

        if result is not None:
            trace1 = go.Scatter(
                # y=np.linspace(-2, 0, Nt),
                # x=result[0],
                y=np.linspace(0, 2, Nt),
                x=result[0][Nt:0:-1],
                mode="lines",
                line=dict(color="pink", width=1),
                name="S<sub>H</sub>",
            )
            trace2 = go.Scatter(
                # y=np.linspace(-2, 0, Nt),
                # x=result[1],
                y=np.linspace(0, 2, Nt),
                x=result[1][Nt:0:-1],
                mode="lines",
                line=dict(color="cyan", width=1),
                name="S<sub>E</sub>",
            )
            trace3 = go.Scatter(
                # y=np.linspace(-2, 0, Nt),
                y=np.linspace(0, 2, Nt),
                x=S0 * np.ones(len(z)),
                line=dict(color="black", width=2, dash="dash"),
                name=f"S<sub>0</sub>: {round(S0,1):g}",
            )
            trace4 = go.Scatter(
                # y=np.linspace(-2, 0, Ns),
                y=np.linspace(0, 2, Nt),
                x=Ssa_h * np.ones(Nt),
                line=dict(color="red", width=2, dash="dash"),
                name=f"S<sub>saH</sub>: {round(Ssa_h,1):g}",
            )
            trace5 = go.Scatter(
                # y=np.linspace(-2, 0, Ns),
                y=np.linspace(0, 2, Nt),
                x=Ssa_e * np.ones(Nt),
                line=dict(color="blue", width=2, dash="dash"),
                name=f"S<sub>saE</sub>: {round(Ssa_e,1):g}",
            )

            layout = go.Layout(
                title=f"TM mode, {frequency:g} MHz, θ = {angle:g}°",
                title_x=0.5,
                xaxis={
                    "title": "S (W/m²)",
                    "range": [0,None],
                },
                yaxis={
                    "title": "height (m)",
                    "range": [0,2],
                },
                width=360,
                height=900,
                template='plotly',
                legend={'yanchor':'top','y':-0.1,
                        'xanchor':'center','x':0.5},
            )

            fig = go.Figure(layout=layout)
            fig.add_trace(trace1)
            fig.add_trace(trace2)
            fig.add_trace(trace3)
            fig.add_trace(trace4)
            fig.add_trace(trace5)
            return fig
        
    elif S0 is None or angle is None or frequency is None:
        raise dash.exceptions.PreventUpdate


@callback(
    Output(component_id="output_graph_TE", component_property="figure"),
    [
        Input("run_button", "n_clicks"),
        Input("ground_radioitem", "value"),
        Input("sa_method_dpdn", "value"),
    ],
    [
        State("S0_input", "value"),
        State("angle_input", "value"),
        State("fMHz_input", "value"),
    ],
)
def update_graph_TE(
    n_clicks, ground_type, sa_method_dpdn, S0, angle, frequency
):
    if (
        n_clicks > 0
        and S0 is not None
        and angle is not None
        and frequency is not None
    ):
        L = 2.0
        ground_type = str(ground_type)
        sa_method_dpdn = str(sa_method_dpdn)
        S0 = float(S0)
        angle = float(angle)
        frequency = float(frequency)
        Ns = compute_ns(frequency, L)
        Nt = Ns + 1
        z = np.linspace(-2, 0, Nt)  # z-direction coordinates
        z[-1] = -1E-6  # make the last z value slightly below zero
        result = compute_power_density(
            ground_type, S0, frequency, angle, "TE", z
        )
        Zs, Ws = sagnd(sa_method_dpdn, Ns, L)
        Ssa_h = np.dot(result[0][0:Nt:1], Ws) / L
        Ssa_e = np.dot(result[1][0:Nt:1], Ws) / L

        if result is not None:
            trace1 = go.Scatter(
                # y=np.linspace(-2, 0, Nt),
                # x=result[0],
                y=np.linspace(0, 2, Nt),
                x=result[0][Nt:0:-1],
                mode="lines",
                line=dict(color="pink", width=1),
                name="S<sub>H</sub>",
            )
            trace2 = go.Scatter(
                # y=np.linspace(-2, 0, Nt),
                # x=result[1],
                y=np.linspace(0, 2, Nt),
                x=result[1][Nt:0:-1],
                mode="lines",
                line=dict(color="cyan", width=1),
                name="S<sub>E</sub>",
            )
            trace3 = go.Scatter(
                # y=np.linspace(-2, 0, Nt),
                y=np.linspace(0, 2, Nt),
                x=S0 * np.ones(len(z)),
                line=dict(color="black", width=2, dash="dash"),
                name=f"S<sub>0</sub>: {round(S0,1):g}",
            )
            trace4 = go.Scatter(
                # y=np.linspace(-2, 0, Ns),
                y=np.linspace(0, 2, Nt),
                x=Ssa_h * np.ones(Nt),
                line=dict(color="red", width=2, dash="dash"),
                name=f"S<sub>saH</sub>: {round(Ssa_h,1):g}",
            )
            trace5 = go.Scatter(
                # y=np.linspace(-2, 0, Ns),
                y=np.linspace(0, 2, Nt),
                x=Ssa_e * np.ones(Nt),
                line=dict(color="blue", width=2, dash="dash"),
                name=f"S<sub>saE</sub>: {round(Ssa_e,1):g}",
            )

            layout = go.Layout(
                title=f"TE mode, {frequency:g} MHz, θ = {angle:g}°",
                xaxis={
                    "showgrid": False,
                    "gridcolor": "black",
                    "title": "Power Flux Density (W/m²)",
                },
                yaxis={
                    "showgrid": False,
                    "gridcolor": "black",
                    "title": "z(m)",
                },
                plot_bgcolor="#fff",
                width=360,
                height=900,
            )
            # fig = go.Figure(data=[trace], layout=layout)
            fig = go.Figure(layout=layout)
            fig.add_trace(trace1)
            fig.add_trace(trace2)
            fig.add_trace(trace3)
            fig.add_trace(trace4)
            fig.add_trace(trace5)
            return fig
    elif S0 is None or angle is None or frequency is None:
        raise dash.exceptions.PreventUpdate

dash.register_page(
    __name__,
    path="/ground",
    title="Ground Reflection",
    name="Ground Reflection",
)

Layout_Ground = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H4(
                            "Input Parameters",
                            style={"color": "Teal", "font-weight": "bold", "font-size":18},
                        ),
                        html.Br(),
                        html.H6(
                            "Ground Type",
                            style={"color": "Teal", "font-weight": "bold", "font-size":14},
                        ),
                        dcc.RadioItems(
                            id="ground_radioitem",
                            options=[
                                {"label": "PEC Ground", "value": "PEC Ground"},
                                {"label": "Wet Soil", "value": "Wet Soil"},
                                {"label": "Dry Soil", "value": "Dry Soil"},
                            ],
                            value="PEC Ground",
                            inline=False,
                        ),
                        html.Br(),
                        html.H6(
                            [
                                "Incident power density, S₀ (W/m²)",
                            ],
                            style={"color": "Teal", "font-weight": "bold", "font-size":14},
                        ),
                        dcc.Input(
                            id="S0_input",
                            type="number",
                            placeholder="",
                            min=0.1,
                            max=1000.0,
                            style={"width": "40%"},
                        ),
                        html.Br(),
                        html.Br(),
                        html.H6(
                            ["Angle of incidence, θ°"],
                            style={"color": "Teal", "font-weight": "bold", "font-size":14},
                        ),
                        dcc.Input(
                            id="angle_input",
                            type="number",
                            placeholder="",
                            min=0.0,
                            max=90.0,  # limit the angle range from 0 to 90degs
                            style={"width": "40%"},
                        ),
                        html.Br(),
                        html.Br(),
                        html.H6(
                            "Frequency (MHz)",
                            style={"color": "Teal", "font-weight": "bold", "font-size":14},
                        ),
                        dcc.Input(
                            id="fMHz_input",
                            type="number",
                            placeholder="",
                            min=1,
                            max=6000,  # limit the freq range from 30MHz to 60GHz
                            style={"width": "40%"},
                        ),
                        html.Br(),
                        html.Br(),
                        html.H6(
                            "Spatial Averaging Method",
                            style={"color": "Teal", "font-weight": "bold", "font-size":14},
                        ),
                        dcc.Dropdown(
                            id="sa_method_dpdn",
                            options=[
                                {
                                    "label": "Simple Spatial Averaging",
                                    "value": "Simple",
                                },
                                {
                                    "label": "Simpson's ⅓ Rule",
                                    "value": "S13",
                                },
                            ],
                            optionHeight=35,  # height/space between dropdown options
                            value="S13",  # dropdown value selected automatically when page loads
                            disabled=False,  # disable dropdown value selection
                            multi=False,  # allow multiple dropdown values to be selected
                            searchable=True,  # allow user-searching of dropdown values
                            search_value="",  # remembers the value searched in dropdown
                            placeholder="Please select...",  # gray, default text shown when no option is selected
                            clearable=False,  # allow user to removes the selected value
                            style={
                                "width": "90%"
                            },  # use dictionary to define CSS styles of your dropdown
                            # className='select_box',           #activate separate CSS document in assets folder
                            # persistence=True,                 #remembers dropdown value. Used with persistence_type
                            # persistence_type='memory'         #remembers dropdown value selected until...
                        ),
                        html.Br(),
                        html.Br(),
                        dbc.Button(
                            "Run",
                            id="run_button",
                            style={"width": "30%"},
                            n_clicks=0,
                        ),
                    ],
                    width={"size": 4},
                ),
                dbc.Col(
                    dcc.Graph(
                        id="output_graph_TM",
                        figure={},
                        style={"box-shadow": "6px 6px 6px lightgrey"},
                    ),
                    width={"size": 4},
                ),
                dbc.Col(
                    dcc.Graph(
                        id="output_graph_TE",
                        figure={},
                        style={"box-shadow": "6px 6px 6px lightgrey"},
                    ),
                    width={"size": 4},
                ),
            ]
        )
    ]
)

layout = dbc.Container(
    [
        html.H4(
            "Power flux density height profile for plane wave obliquely incident on PEC or real ground",
            style={
                "color": "Teal",
                "font-weight": "bold",
                "textAlign": "center",
            },
        ),
        html.Hr(),
        Layout_Ground,
    ]
)
