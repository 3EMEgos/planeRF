import dash
import dash_bootstrap_components as dbc
from dash import html, dcc

dash.register_page(__name__, path="/", title="EMF App Hub", name="Home")

dash.page_container.style = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "1rem 1rem",
}

card_main1 = dbc.Card(
    [
        dbc.CardImg(
            src="/assets/image-1.png", top=True, bottom=False, title="Image by Vitas"
        ),
        dbc.CardBody(
            [
                html.H5("EMF APP-1", className="card-title"),
                # html.H6("APP 1:", className="card-subtitle"),
                html.P(
                    "Plane Wave Incident onto PEC or Real Ground Plane",
                    className="card-text",
                ),
                # dbc.Button("Press me", color="primary"),
                dbc.CardLink(
                    "Ground Plane Reflection", href="/ground", target="_blank"
                ),
            ]
        ),
    ],
    color="dark",
    inverse=True,  # change color of text (black or white)
    outline=True,  # True = remove the block colors from the background and header
)

card_main2 = dbc.Card(
    [
        dbc.CardImg(
            src="/assets/image-2.png", top=True, bottom=False, title="Image by Vitas"
        ),
        dbc.CardBody(
            [
                html.H5("EMF APP-2", className="card-title"),
                # html.H6("APP 2:", className="card-subtitle"),
                html.P(
                    "Plane Wave Incident onto Human Tissues, Check SAR",
                    className="card-text",
                ),
                # dbc.Button("Press me", color="primary"),
                dbc.CardLink("Human Tissue SAR", href="/tissue", target="_blank"),
            ]
        ),
    ],
    color="dark",
    inverse=True,  # change color of text (black or white)
    outline=True,  # True = remove the block colors from the background and header
)

layout = dbc.Container(
    [
        html.H4(
            "The three musketeers of online RF dosimetry",
            style={"textAlign": "center", "margin-bottom": "0px"},
        ),
        html.Hr(),
        dbc.Row(
            [dbc.Col(card_main1, width=3), dbc.Col(card_main2, width=3)],
            justify="around",
        ),
    ]
)
