import dash
import dash_bootstrap_components as dbc
from dash import html, dcc

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY], use_pages=True)

# styling the sidebar
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "1rem 1rem",
    # "background-color": "#f8f9fa",
}

# padding for the page content
CONTENT_STYLE = {
    "margin-left": "18rem",  # "18rem",
    "margin-right": "2rem",  # "2rem",
    "padding": "1rem 1rem",
}

Sidebar = html.Div(
    [
        html.Hr(),
        html.H4("planeEME app", style={"textAlign": "center"}),
        html.Hr(),
        dbc.Nav(
            [
                dbc.NavLink("Home", href="/", active="exact"),
                dbc.NavLink("Ground Plane Reflection", href="/ground", active="exact"),
                dbc.NavLink("Human Tissue SAR", href="/tissue", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)

Content = html.Div(id="container", children=[], style=CONTENT_STYLE)

app.layout = html.Div([dcc.Location(id="url"), Sidebar, Content, dash.page_container])

if __name__ == "__main__":
    app.run_server(debug=True, port=3000)
