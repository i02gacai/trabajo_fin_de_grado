import base64
import io
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
import plotly.express as plx
import matplotlib.pyplot as plt
import pandas as pd
from dash.dependencies import Input, Output, State
from app import app
import numpy as np
from scipy.stats import entropy
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import problexity as px

def t(key, lang):
    return lang.get(key, key)

def layout(lang):
    return dbc.Container(
        [
            html.Hr(),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Label(t("Select-Input-Feature", lang)),
                            dcc.Dropdown(id='input-feature-dropdown_', multi=True),
                        ],
                        width=6,
                    ),
                    dbc.Col(
                        [
                            html.Label(t("Select-Class-Feature", lang)),
                            dcc.Dropdown(id='class-feature-dropdown_', multi=False),
                        ],
                        width=6,
                    ),
                ]
            ),
            html.Br(),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Button(t("Calculate-Measures", lang), id="calculate-measures", n_clicks=0),
                        ],
                        width=12,
                        className="text-center",
                    ),
                ],
            ),
            html.Br(),
            html.Div(id='error-message_'), 
            html.Div(id='complexity-report'),
            html.Img(id="output-image", style={'height': '400px', 'width': '50%', 'margin-left': '25%'}, hidden=True),
            html.Br(),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Button(t("Download-CSV", lang), id="download-csv_", n_clicks=0, hidden=True),
                        ],
                        width=12,
                        className="text-center",
                    ),
                ],
            ),
            dcc.Download(id="download-component_"),
        ],
        fluid=True,
    )


    