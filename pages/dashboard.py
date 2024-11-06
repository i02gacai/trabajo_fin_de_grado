import base64
import io
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
import plotly.express as plx
import pandas as pd
from dash.dependencies import Input, Output, State
import numpy as np
from app import app

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
                            dcc.Dropdown(id='input-feature-dropdown', multi=True),
                        ],
                        width=6,
                    ),
                    dbc.Col(
                        [
                            html.Label(t("Select-Class-Feature", lang)),
                            dcc.Dropdown(id='class-feature-dropdown', multi=False),
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
                            html.Label(t("Select-Graph-Type", lang)),
                            dcc.Dropdown(
                                id='graph-type-dropdown',
                                options=[
                                    {"label": t("Scatter-Plot", lang), "value": "scatter"},
                                    {"label": t("Pie-Chart", lang), "value": "pie"},
                                    {"label": t("Histogram", lang), "value": "histogram"},
                                    {"label": t("Box-Plot", lang), "value": "box"},
                                    {"label": t("Heatmap", lang), "value": "heatmap"},
                                ],
                                value="scatter",
                            ),
                        ],
                        width=6,
                    ),
                ],
            ),
            html.Br(),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Button(t("Generate-Plot", lang), id="generate-plot", n_clicks=0),
                        ],
                        width=12,
                        className="text-center",
                    ),
                ],
            ),
            html.Br(),
            html.Div(id = "error-message-plot"),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dcc.Graph(id="output-graph"),
                        ],
                        width=12,
                    ),
                ],
            ),
        ],
        fluid=True,
    )


@app.callback(
    Output("input-feature-dropdown", "options"),
    Output("class-feature-dropdown", "options"),
    Input("stored-data2", "data"),
    
)
def update_dropdown_options(data):
    if data:
        df = pd.DataFrame(data)
        options = [{"label": col, "value": col} for col in df.columns]
        return options, options
    return [], []

@app.callback(
    [Output("output-graph", "figure"),
     Output('error-message-plot', 'children')],
    [Input("generate-plot", "n_clicks"),
    Input("language", "data")],
    State("stored-data2", "data"),
    State("input-feature-dropdown", "value"),
    State("class-feature-dropdown", "value"),
    State("graph-type-dropdown", "value"),
)
def update_graph(n_clicks, lang, data, input_feature, class_feature, graph_type):
    if n_clicks > 0:
        if not input_feature or not class_feature or not graph_type:
            return {}, dbc.Alert(t("Plot-Alert", lang), color="danger")
        
        if data:
            df = pd.DataFrame(data)

            # Manejo de gráficos
            if graph_type == "scatter":
                fig = plx.scatter(df, x=input_feature[0], color=class_feature, title=f'{input_feature[0]} vs {class_feature}')
            elif graph_type == "bar":
                fig = plx.bar(df, x=input_feature[0], color=class_feature, title=f'{input_feature[0]} vs {class_feature}')
            elif graph_type == "pie":
                fig = plx.pie(df, names=class_feature, title=f'Distribution of {class_feature}')
            elif graph_type == "histogram":
                fig = plx.histogram(df, x=input_feature[0], color=class_feature, title=f'Distribution of {input_feature[0]}')
            elif graph_type == "line":
                fig = plx.line(df, x=input_feature[0], y=input_feature[1], color=class_feature, title=f'{input_feature[0]} vs {input_feature[1]}')
            elif graph_type == "box":
                fig = plx.box(df, x=class_feature, y=input_feature[0], title=f'Box Plot of {input_feature[0]} by {class_feature}')
            elif graph_type == "heatmap":
                correlation = df.corr()  # Calcular la correlación
                fig = plx.imshow(correlation, title='Heatmap of Correlation')
            elif graph_type == "area":
                fig = plx.area(df, x=input_feature[0], y=input_feature[1], title=f'Area Chart of {input_feature[0]} and {input_feature[1]}')
            
            return fig, html.Div()  # No mostrar mensaje de error cuando todo está bien

    return {}, html.Div()