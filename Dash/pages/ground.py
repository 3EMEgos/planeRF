import dash
from dash import dcc, callback, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objs as go

from planeRF import compute_power_density


@callback(
    Output(component_id="output-graphTM", component_property="figure"),
    [Input("run-button", "n_clicks"), Input("Ground_raditem", "value")],
    [
        State("efield-input", "value"),
        State("angle-input", "value"),
        State("frequency-input", "value"),
    ],
)
def update_graph_TM(n_clicks, ground_type, E0, angle, frequency):
    if n_clicks > 0 and E0 is not None and angle is not None and frequency is not None:
        ground_type = str(ground_type)
        E0 = float(E0)
        angle = float(angle)
        frequency = float(frequency)
        result = compute_power_density(ground_type, E0, frequency, angle, "TM")
        if result is not None:
            trace1 = go.Scatter(
                y=np.linspace(-2, 0, 2001), x=result[0], mode="lines", name="SH"
            )
            trace2 = go.Scatter(
                y=np.linspace(-2, 0, 2001), x=result[1], mode="lines", name="SE"
            )
            trace3 = go.Scatter(
                y=np.linspace(-2, 0, 2001),
                x=result[2],
                line=dict(color="lime", width=2, dash="dash"),
                name="S0",
            )

            layout = go.Layout(
                title=f"TM mode, {frequency:g} MHz, theta = {angle:g}°",
                xaxis=dict(title="Power Flux Density (W/m2)"),
                yaxis=dict(title="z (m)"),
            )

            # fig = go.Figure(data=[trace], layout=layout)
            fig = go.Figure(layout=layout)
            fig.add_trace(trace1)
            fig.add_trace(trace2)
            fig.add_trace(trace3)
            return fig
    elif angle is None or frequency is None:
        raise dash.exceptions.PreventUpdate


@callback(
    Output(component_id="output-graphTE", component_property="figure"),
    [Input("run-button", "n_clicks"), Input("Ground_raditem", "value")],
    [
        State("efield-input", "value"),
        State("angle-input", "value"),
        State("frequency-input", "value"),
    ],
)
def update_graph_TE(n_clicks, ground_type, E0, angle, frequency):
    if n_clicks > 0 and E0 is not None and angle is not None and frequency is not None:
        ground_type = str(ground_type)
        E0 = float(E0)
        angle = float(angle)
        frequency = float(frequency)
        result = compute_power_density(ground_type, E0, frequency, angle, "TE")
        if result is not None:
            trace1 = go.Scatter(
                y=np.linspace(-2, 0, 2001), x=result[0], mode="lines", name="SH"
            )
            trace2 = go.Scatter(
                y=np.linspace(-2, 0, 2001), x=result[1], mode="lines", name="SE"
            )
            trace3 = go.Scatter(
                y=np.linspace(-2, 0, 2001),
                x=result[2],
                line=dict(color="lime", width=2, dash="dash"),
                name="S0",
            )

            layout = go.Layout(
                title=f"TE mode, {frequency:g} MHz, theta = {angle:g}°",
                xaxis=dict(title="Power Flux Density (W/m2)"),
                yaxis=dict(title="z (m)"),
            )
            # fig = go.Figure(data=[trace], layout=layout)
            fig = go.Figure(layout=layout)
            fig.add_trace(trace1)
            fig.add_trace(trace2)
            fig.add_trace(trace3)
            return fig
    elif angle is None or frequency is None:
        raise dash.exceptions.PreventUpdate


dash.register_page(
    __name__,
    path="/ground",
    title="Ground Reflection",
    name="Ground Plane Reflection",
)

# dash.page_container.style = {
#     "margin-left": "18rem",
#     "margin-right": "2rem",
#     "padding": "1rem 1rem",
# }

Layout_Ground = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H4(
                            "Input Parameters",
                            style={"color": "Teal", "font-weight": "bold"},
                        ),
                        html.Br(),
                        html.H6(
                            "Ground Type",
                            style={"color": "Teal", "font-weight": "bold"},
                        ),
                        dcc.RadioItems(
                            id="Ground_raditem",
                            options=[
                                {"label": "PEC Ground", "value": "PEC Ground"},
                                {"label": "Real Ground", "value": "Real Ground"},
                            ],
                            value="PEC Ground",
                            inline=False,
                        ),
                        html.Br(),
                        html.H6(
                            ["Incident e-field Strengh (V/m)"],
                            style={"color": "Teal", "font-weight": "bold"},
                        ),
                        dcc.Input(
                            id="efield-input",
                            type="number",
                            placeholder="",
                            style={"width": "40%"},
                        ),
                        html.Br(),
                        html.Br(),
                        html.H6(
                            ["Angle of Incident (", html.Sup("o"), ")"],
                            style={"color": "Teal", "font-weight": "bold"},
                        ),
                        dcc.Input(
                            id="angle-input",
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
                            style={"color": "Teal", "font-weight": "bold"},
                        ),
                        dcc.Input(
                            id="frequency-input",
                            type="number",
                            placeholder="",
                            min=1,
                            max=6000,  # limit the freq range from 30MHz to 60GHz
                            style={"width": "40%"},
                        ),
                        html.Br(),
                        html.Br(),
                        dbc.Button(
                            "Run", id="run-button", style={"width": "30%"}, n_clicks=0
                        ),
                    ],
                    width={"size": 4},
                ),
                dbc.Col(
                    dcc.Graph(
                        id="output-graphTM",
                        figure={},
                        style={"box-shadow": "6px 6px 6px lightgrey"},
                    ),
                    width={"size": 4},
                ),
                dbc.Col(
                    dcc.Graph(
                        id="output-graphTE",
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
            "Plane Wave Oblique Incident onto PEC or Real Ground: TE and TM",
            style={"color": "Teal", "font-weight": "bold", "textAlign": "center"},
        ),
        html.Hr(),
        Layout_Ground,
    ]
)
