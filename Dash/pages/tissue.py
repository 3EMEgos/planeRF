import dash
import dash_bootstrap_components as dbc
from dash import html


dash.register_page(
    __name__, path="/tissue", title="Human Tissue SAR", name="TissueSAR"
)
dash.page_container.style = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "1rem 1rem",
}
layout = dbc.Container(
    [
        html.H4(
            "Plane Wave Incident onto Human Tissue (SAR)",
            style={"textAlign": "center", "margin-bottom": "0px"},
        ),
        html.Hr(),
        html.P("It is currently under development"),
    ]
)
