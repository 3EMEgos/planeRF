import dash
from dash import html, dcc, Dash
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template

# Create dash app with a FLATLY theme
app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY], use_pages=True)

# Apply FLATLY theme to figures
load_figure_template("FLATLY")


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

Sidebar = html.Div(
    [
        html.Hr(),
        html.H4("planeEME app", style={"textAlign": "center"}),
        html.Hr(),
        dbc.Nav(
            [
                dbc.NavLink("Home", href="/", active="exact"),
                dbc.NavLink("Ground Reflection", href="/ground", active="exact"),
                dbc.NavLink("Human Tissue SAR", href="/tissue", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)

Content = html.Div(id="container", children=[])

app.layout = html.Div([dcc.Location(id="url"), Sidebar, Content, dash.page_container])

if __name__ == "__main__":
    app.run_server(debug=True, port=3000)
