import dash
from dash import dcc, callback, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objs as go
from .common import compute_S_params, compute_power_density, sagnd, compute_ns


@callback(
    Output(component_id="output_graph_TM", component_property="figure"),
    [
        Input("ground_radioitem", "value"),
        Input("S0_input", "value"),
        Input("fMHz_input", "value"),
        Input("mode_radioitem", "value"),
        Input("sa_method_dpdn", "value"),
        Input("L_input", "value"),
        Input("Nsap_input", "value"),
        Input("angle_input", "value"),
    ],
)
def update_graph(gnd, S0, fMHz, pol, sa_method, L, Nsap, theta):
    if (
        gnd is not None
        and S0 is not None
        and fMHz is not None
        and pol is not None
        and sa_method is not None
        and L is not None
        and Nsap is not None
        and theta is not None
    ):
        # Calculate S computational parameters
        gnd = str(gnd)
        sa_method = str(sa_method)
        S0 = float(S0)
        theta = float(theta)
        fMHz = float(fMHz)
        S_params, (epsr, sigma) = compute_S_params(gnd, fMHz, theta, pol)

        # Calculate S plot
        Ns = compute_ns(fMHz, L)  # No. of S sampling points
        z = np.linspace(0, -2, Ns)  # z coords for sampling points
        z[0] = -1e-12  # make the first z value slightly above zero
        result = compute_power_density(z, S0, *S_params)
        SH, SE = result

        # Calculate spatial averaging points and averages
        # Note that compute_power_density can change Nsap if it is not valid for the spatial averaging method
        Zsap, Wsap, Nsap = sagnd(sa_method, Nsap, L)
        SHsap, SEsap = compute_power_density(Zsap, S0, *S_params)
        SHsa = np.dot(SHsap, Wsap)
        SEsa = np.dot(SEsap, Wsap)

        # Calculate z & W for ACCURATE ESTIMATE of spatial averaging points
        Nsap_accurate = 200  # Set this to a very high number for good accuracy
        sa_method_accurate = "GQR"  # Use GQR for go0d accuracy
        Zsap_acc, Wsap_acc, Nsap_acc = sagnd(sa_method_accurate, Nsap_accurate, L)
        SHsap_acc, SEsap_acc = compute_power_density(Zsap_acc, S0, *S_params)
        SHsa_accurate = np.dot(SHsap_acc, Wsap_acc)
        SEsa_accurate = np.dot(SEsap_acc, Wsap_acc)

        # Calculate percentages of spatial average estimates to accurate estimates
        SHsa_pc = (SHsa / SHsa_accurate) * 100.0
        SEsa_pc = (SEsa / SEsa_accurate) * 100.0

        # Create the plot traces
        Y = np.linspace(0, 2, Ns)  # make z levels positive for plotting
        if (result) is not None:
            trace1_SE = go.Scatter(
                y=Y,
                x=SE,
                mode="lines",
                line=dict(color="#EE6677", width=1.5),  # red EE6677
                name="S<sub>E</sub>",
            )
            trace2_SH = go.Scatter(
                y=Y,
                x=SH,
                mode="lines",
                line=dict(color="#66CCEE", width=1.5),  # cyan 66CCEE
                name="S<sub>H</sub>",
            )
            trace3_S0 = go.Scatter(
                y=[0, 2],
                x=[S0, S0],
                mode="lines",
                line=dict(color="#228833", width=1.5),  # green 228833
                name=f"S<sub>0</sub> = {round(S0,2):g}",
            )
            trace4_SEsa = go.Scatter(
                y=[0, 2],
                x=[SEsa, SEsa],
                mode="lines",
                line=dict(color="#AA3377", width=1.5, dash="dashdot"),  # purple AA3377
                name=f"S<sub>E sa</sub> = {round(SEsa,2):g}  ({SEsa_pc:0.1f}%)",
            )
            trace5_SHsa = go.Scatter(
                y=[0, 2],
                x=[SHsa, SHsa],
                mode="lines",
                line=dict(color="#4477AA", width=1.5, dash="dash"),  # blue 4477AA
                name=f"S<sub>H sa</sub> = {round(SHsa,2):g}  ({SHsa_pc:0.1f}%)",
            )
            trace5_SEsap = go.Scatter(
                y=-Zsap,
                x=SEsap,
                mode="markers",
                marker=dict(color="#AA3377", size=8),  # purple AA3377
                name="S<sub>E</sub> spatial averaging points",
            )
            trace6_SHsap = go.Scatter(
                y=-Zsap,
                x=SHsap,
                mode="markers",
                marker=dict(color="#4477AA", size=8),  # blue 4477AA
                name="S<sub>H</sub> spatial averaging points",
            )

            # Generate plot title
            sigma[1] = round(sigma[1], 10)  # round very small numbers to zero
            t1 = f"<b>{gnd}</b> "
            t2 = f'<span style="font-size:12pt">(ε<sub>r</sub>={epsr[1]:0.2g}, σ={sigma[1]:0.3g} S/m)</span>'
            t3 = f"<b>{fMHz:g} MHz,  θ = {theta:g}°,  {pol} mode</b>"
            if sa_method == "PS":
                t4 = f'<span style="font-size:12pt">Point spatial estimate at {L}m</span>'
            else:
                t4 = f'<span style="font-size:12pt">{sa_method} averaging for {Nsap} points over {L}m</span>'
            plot_title = t1 + t2 + "<br>" + t3 + "<br>" + t4

            layout = go.Layout(
                title=plot_title,
                margin_t=150,  # height for plot title
                title_x=0.5,
                xaxis={
                    "title": "S (W/m²)",
                    "range": [0, 4.1 * S0],
                },
                yaxis={
                    "title": "height above ground (m)",
                    "range": [0, 2.02],
                },
                width=350,
                height=800,
                legend={"yanchor": "top", "y": -0.1, "xanchor": "center", "x": 0.5},
            )

            # Add plot traces to the figure
            fig = go.Figure(layout=layout)
            fig.add_trace(trace1_SE)
            fig.add_trace(trace2_SH)
            fig.add_trace(trace3_S0)
            fig.add_trace(trace5_SEsap)
            fig.add_trace(trace4_SEsa)
            fig.add_trace(trace6_SHsap)
            fig.add_trace(trace5_SHsa)

            return fig

    elif S0 is None or theta is None or fMHz is None:
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
                # LEFT COLUMN
                dbc.Col(
                    [
                        # INPUT PARAMETERS HEADING
                        html.H4(
                            "Input Parameters",
                            style={
                                "color": "Teal",
                                "font-weight": "bold",
                                "font-size": 18,
                                "text-align": "center",
                            },
                        ),
                        html.Br(),
                        # GROUND TYPE INPUTS
                        html.H6(
                            "Ground Type",
                            style={
                                "color": "Teal",
                                "font-weight": "bold",
                                "font-size": 14,
                            },
                        ),
                        dcc.RadioItems(
                            id="ground_radioitem",
                            options=[
                                {"label": " No ground (Air)", "value": "Air"},
                                {"label": " Metal (PEC) ground", "value": "PEC Ground"},
                                {"label": " Wet soil", "value": "Wet Soil"},
                                {
                                    "label": " Medium dry ground",
                                    "value": "Medium Dry Ground",
                                },
                                {"label": " Dry soil", "value": "Dry Soil"},
                                {"label": " Concrete", "value": "Concrete"},
                            ],
                            value="PEC Ground",
                            inline=False,
                        ),
                        html.Br(),
                        # S0 INPUT
                        html.H6(
                            [
                                "Incident power density, S₀ (W/m²)",
                            ],
                            style={
                                "color": "Teal",
                                "font-weight": "bold",
                                "font-size": 14,
                            },
                        ),
                        dcc.Input(
                            id="S0_input",
                            type="number",
                            placeholder="",
                            min=0,
                            max=1000.0,
                            value=100,
                            # step=10,
                            style={"width": "40%"},
                        ),
                        html.Br(),
                        html.Br(),
                        # THETA INPUT
                        html.H6(
                            ["Angle of incidence, θ°"],
                            style={
                                "color": "Teal",
                                "font-weight": "bold",
                                "font-size": 14,
                            },
                        ),
                        dcc.Input(
                            id="angle_input",
                            type="number",
                            placeholder="",
                            min=0.0,
                            max=90.0,  # limit the angle range from 0 to 90degs
                            value=30,
                            style={"width": "40%"},
                        ),
                        html.Br(),
                        html.Br(),
                        # fMHZ INPUT
                        html.H6(
                            "Frequency (MHz)",
                            style={
                                "color": "Teal",
                                "font-weight": "bold",
                                "font-size": 14,
                            },
                        ),
                        dcc.Input(
                            id="fMHz_input",
                            type="number",
                            placeholder="",
                            min=30,
                            max=6000,  # limit the freq range from 30MHz to 60GHz
                            value=900,
                            style={"width": "40%"},
                        ),
                        html.Br(),
                        html.Br(),
                        # POLARISATION MODE INPUT
                        html.H6(
                            "Polarisation mode",
                            style={
                                "color": "Teal",
                                "font-weight": "bold",
                                "font-size": 14,
                            },
                        ),
                        dcc.RadioItems(
                            id="mode_radioitem",
                            options=[
                                {"label": " TM", "value": "TM"},
                                {"label": " TE", "value": "TE"},
                            ],
                            value="TM",
                            inline=False,
                        ),
                        html.Br(),
                        # SPATIAL AVERAGING METHOD INPUT
                        html.H6(
                            "Spatial Averaging Method",
                            style={
                                "color": "Teal",
                                "font-weight": "bold",
                                "font-size": 14,
                            },
                        ),
                        dcc.Dropdown(
                            id="sa_method_dpdn",
                            options=[
                                {"label": "Point spatial", "value": "PS"},
                                {"label": "Simple Averaging", "value": "Simple"},
                                {"label": "Riemann Sum", "value": "RS"},
                                {"label": "Simpson's ⅓ Rule", "value": "S13"},
                                {"label": "Simpson's ⅜ Rule", "value": "S38"},
                                {"label": "Gaussian Quadrature Rule", "value": "GQR"},
                            ],
                            optionHeight=35,  # height/space between dropdown options
                            value="Simple",  # dropdown value selected automatically when page loads
                            disabled=False,  # disable dropdown value selection
                            multi=False,  # allow multiple dropdown values to be selected
                            searchable=True,  # allow user-searching of dropdown values
                            search_value="",  # remembers the value searched in dropdown
                            placeholder="Please select...",  # gray, default text shown when no option is selected
                            clearable=False,  # allow user to removes the selected value
                            style={
                                "width": "95%",
                                "font-size": 12,
                            },  # use dictionary to define CSS styles of your dropdown
                            # className='select_box',           #activate separate CSS document in assets folder
                            # persistence=True,                 #remembers dropdown value. Used with persistence_type
                            # persistence_type='memory'         #remembers dropdown value selected until...
                        ),
                        html.Br(),
                        # SPATIAL AVERAGING LENGTH, L, INPUT
                        html.H6(
                            "Spatial averaging length",
                            style={
                                "color": "Teal",
                                "font-weight": "bold",
                                "font-size": 14,
                            },
                        ),
                        dcc.Input(
                            id="L_input",
                            type="number",
                            placeholder="",
                            min=0.3,
                            max=2,  # limit the freq range from 30MHz to 60GHz
                            value=1.6,
                            step=0.1,
                            style={"width": "40%"},
                        ),
                        html.Br(),
                        html.Br(),
                        # SPATIAL AVERAGING NO. OF POINTS, Nsap, INPUT
                        html.H6(
                            "No. of spatial averaging points",
                            style={
                                "color": "Teal",
                                "font-weight": "bold",
                                "font-size": 14,
                            },
                        ),
                        dcc.Input(
                            id="Nsap_input",
                            type="number",
                            placeholder="",
                            min=1,
                            max=300,  # limit the freq range from 30MHz to 60GHz
                            value=7,
                            step=1,
                            style={"width": "40%"},
                        ),
                    ],
                    width={"size": 4},
                ),
                # RIGHT COLUMN
                dbc.Col(
                    [
                        # 1st ROW
                        dbc.Row(
                            [
                                # 1st COLUMN OF 1st ROW
                                dbc.Col(
                                    # S PLOT
                                    dcc.Graph(
                                        id="output_graph_TM",
                                        figure={},
                                        # style={"box-shadow": "6px 6px 6px lightgrey"},
                                    ),
                                    width={"size": 6},
                                ),
                                # 2nd COLUMN OF 1st ROW
                                dbc.Col(
                                    # CODE for dynamic pictograph
                                    # of incident S0 ray showing theta
                                ),
                            ],
                        ),
                    ],
                ),
            ]
        )
    ]
)

style_H4 = {
    "color": "Teal",
    "font-weight": "bold",
    "textAlign": "center",
}
layout = dbc.Container(
    [
        html.H4(
            "Power flux density height profile for plane wave",
            style=style_H4,
        ),
        html.H4(
            "obliquely incident on PEC or real ground",
            style=style_H4,
        ),
        html.Hr(),
        Layout_Ground,
    ]
)
