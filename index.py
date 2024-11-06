import base64
import io
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from flask import request, g
from scipy.io import arff
from scipy.stats import entropy
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from pymfe.mfe import MFE
import dash
from dash import dcc, html, dash_table
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import plotly.express as plx
import problexity as px
import pages
from app import app


# The way to run the app in a local server.
app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.BOOTSTRAP, "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css"])
server = app.server

initial_state = False

# Definir la función para cargar el archivo de idioma
def load_language(lang_code):
    try:
        with open(f'locales/{lang_code}.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        # En caso de que el archivo de idioma no exista, cargamos inglés por defecto
        with open('locales/en.json', 'r', encoding='utf-8') as f:
            return json.load(f)

@app.server.before_request
def set_language():
    user_language = request.headers.get('Accept-Language', 'en').split(',')[0]
    g.lang_code = user_language[:2]  # Obtiene el código del idioma, por ejemplo, 'es' para español
    g.lang = load_language(g.lang_code)

# Definir la función de traducción
def t(key, lang):
    return lang.get(key, key)  # Si no encuentra la clave, devuelve la clave misma

# Función para obtener archivos en la carpeta "datasets"
def get_dataset_files():
    dataset_folder = 'dataset'
    if os.path.exists(dataset_folder):
        return [f for f in os.listdir(dataset_folder) if os.path.isfile(os.path.join(dataset_folder, f))]
    return []

app.layout = dbc.Container(
    [   
        # A component that allows to get the current URL of the page.
        dcc.Location(id="url", refresh=False),
        dcc.Store(id="language", data=None, storage_type='session'),
         # Components that allow to store data in the whole the app.
        dcc.Store(id='stored-data', data=None, storage_type='session'),
        dcc.Store(id='stored-data2', data=None, storage_type='session'),
        dcc.Download(id="download-component-index"),
        dcc.Input(id='trigger', value='', type='text', style={'display': 'none'}),
        html.Div(id='output')
    ]
)

@app.callback(
    Output('output', 'children', allow_duplicate=True),
    Output('language', 'data',  allow_duplicate=True),
    Input('trigger', 'value'),
    prevent_initial_call='initial_duplicate'
)
def initial_update_output(trigger):
    global initial_state
    if not initial_state:
        initial_state = True
        return html.Div(
            [   
                # Creating the navbar of the app.
                dbc.Navbar(
                    children=[
                        html.A(
                            dbc.Row(
                                [
                                    dbc.Col(
                                        # Creating an image with the logo of the app.
                                        html.Img(
                                            src=app.get_asset_url("logo.png"), height="80px"
                                        )
                                    ),
                                ],
                                className="g-0 ml-auto flex-nowrap mt-3 mt-md-0",
                                align="center",
                            ),
                            href=app.get_relative_path("/"),
                        ),
                        # Creating the items of the navbar.
                        dbc.Row(
                            [
                                dbc.Collapse(
                                    dbc.Nav(
                                        [
                                            # The pages of the app.
                                            dbc.NavItem(dbc.NavLink(t("Home",g.lang), href=app.get_relative_path("/"))),
                                            dbc.NavItem(dbc.NavLink(t("Dashboard",g.lang), href=app.get_relative_path("/dash"))),
                                            dbc.NavItem(dbc.NavLink(t("Meta-Feature",g.lang), href=app.get_relative_path("/metafeatures"))),
                                            dbc.NavItem(dbc.NavLink(t("Complexity",g.lang), href=app.get_relative_path("/complexity"))),
                                        ],
                                        className="w-100",
                                        fill=True,
                                        horizontal='end'
                                    ),
                                    navbar=True,
                                    is_open=True,
                                ),
                            ],
                            className="flex-grow-1",
                        ),
                        dbc.Row(
                                [
                                    html.I(className="fas fa-language fa-fw mr-1"),
                                    dcc.Dropdown(
                                        id='language-dropdown',
                                        options=[
                                            {'label': 'English (en)', 'value': 'en'},
                                            {'label': 'Spanish (es)', 'value': 'es'},
                                            # Agregar más idiomas aquí si es necesario
                                        ],
                                        value=g.lang_code,  # Valor inicial basado en el idioma actual
                                        clearable=False,
                                        style = { 'width': '150px', 
                                            'margin-left': '5px',}
                                    ),
                                ],
                                id = 'language-row',
                                className="g-0 ml-auto flex-nowrap mt-3 mt-md-0",
                                align="center",
                                style = {   'width': '200px',        # Ancho del dropdown
                                                    'height': '50px',
                                                    'backgroundColor': '#f9f9f9',
                                                    'margin-top': '20px',
                                                    'margin-left': '5px',}
                            ),
                    ],
                ),
                # BODY
                dbc.Row(
                    [
                        dbc.Col(
                            html.Div(
                                id='dropdown-div',
                                children = [ 
                                    dcc.Dropdown(
                                    id='dataset-dropdown',
                                    options=[{'label': f, 'value': f} for f in get_dataset_files()],
                                    placeholder=t("Select-dataset",g.lang),
                                ),
                                ]
                            ),
                            width=6,
                        ),
                        dbc.Col(
                            html.Div(
                                id='upload-div',
                                children=[
                                    dcc.Upload(
                                        id="upload-data",
                                        children=html.Div([t("Drag-drop",g.lang)]),
                                        style={
                                            "width": "100%",
                                            "height": "60px",
                                            "lineHeight": "60px",
                                            "border" : "#007bff",
                                            "borderWidth": "1px",
                                            "borderStyle": "dashed",
                                            "borderRadius": "5px",
                                            "textAlign": "center",
                                            "margin": "10px",
                                        },
                                        multiple=False,
                                    ),
                                    html.Br(),
                                ]
                            ),
                            width=6,
                        )
                    ],
                    align="center",  # Vertically align the components in the center
                    className="mb-4",  # Add some bottom margin for spacing
                ),
                
                # A component that shows a message to the user about the data file.
                dbc.Alert([t("Msg-load-file",g.lang)],
                          id="alert-auto",
                          is_open=False,
                          dismissable=True,
                          fade=True,
                          ),

                html.Div( id = "card",
                    children=[
                        dbc.Card(
                            children=[
                                # A card with information about the web aplication.
                                dbc.CardHeader(t("Welcome", g.lang)),
                                dbc.CardBody(
                                    [
                                        dcc.Markdown(
                                            t("welcome-card",g.lang),
                                            style={"margin": "0 10px"},
                                        )
                                    ]
                                ),
                            ]
                        ),
                        html.Br(),
                    ],                
                ),

                html.Div(id="page-content"),
                html.Div(id="error-message"),
                html.Div(id="table-div"),  
                
                dbc.Row(
                    [
                        dbc.Col(
                            [   
                                html.Br(),
                                html.Button(t("Download-CSV", g.lang), id="download-csv-index", n_clicks=0, hidden = True),
                            ],
                            width=12,
                            className="text-center",
                        ),
                    ],
                ),
                ]
        ), g.lang
    return dash.no_update, dash.no_update  # Si ya se inicializó, no actualizar el contenido

@app.callback(
    Output('output', 'children', allow_duplicate=True),
    Output('language', 'data', allow_duplicate=True),
    Input('language-dropdown', 'value'),
    prevent_initial_call=True
)
def update_output(value):
    lang = load_language(value)
    return html.Div(
        [   
            # Creating the navbar of the app.
            dbc.Navbar(
                children=[
                    html.A(
                        dbc.Row(
                            [
                                dbc.Col(
                                    # Creating an image with the logo of the app.
                                    html.Img(
                                        src=app.get_asset_url("logo.png"), height="80px"
                                    )
                                ),
                            ],
                            className="g-0 ml-auto flex-nowrap mt-3 mt-md-0",
                            align="center",
                        ),
                        href=app.get_relative_path("/"),
                    ),
                    # Creating the items of the navbar.
                    dbc.Row(
                        [
                            dbc.Collapse(
                                dbc.Nav(
                                    [
                                        # The pages of the app.
                                        dbc.NavItem(dbc.NavLink(t("Home",lang), href=app.get_relative_path("/"))),
                                        dbc.NavItem(dbc.NavLink(t("Dashboard",lang), href=app.get_relative_path("/dash"))),
                                        dbc.NavItem(dbc.NavLink(t("Meta-Feature",lang), href=app.get_relative_path("/metafeatures"))),
                                        dbc.NavItem(dbc.NavLink(t("Complexity",lang), href=app.get_relative_path("/complexity"))),
                                    ],
                                    className="w-100",
                                    fill=True,
                                    horizontal='end'
                                ),
                                navbar=True,
                                is_open=True,
                            ),
                        ],
                        className="flex-grow-1",
                    ),
                    dbc.Row(
                            [
                                html.I(className="fas fa-language fa-fw mr-1"),
                                dcc.Dropdown(
                                    id='language-dropdown',
                                    options=[
                                        {'label': 'English (en)', 'value': 'en'},
                                        {'label': 'Spanish (es)', 'value': 'es'},
                                        # Agregar más idiomas aquí si es necesario
                                    ],
                                    value= value,  # Valor inicial basado en el idioma actual
                                    clearable=False,
                                    style = { 'width': '150px', 
                                        'margin-left': '5px',}
                                ),
                            ],
                            id = 'language-row',
                            className="g-0 ml-auto flex-nowrap mt-3 mt-md-0",
                            align="center",
                            style = {   'width': '200px',        # Ancho del dropdown
                                                'height': '50px',
                                                'backgroundColor': '#f9f9f9',
                                                'margin-top': '20px',
                                                'margin-left': '5px',}
                        ),
                ],
            ),
            # BODY
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(
                            id='dropdown-div',
                            children = [ 
                                dcc.Dropdown(
                                id='dataset-dropdown',
                                options=[{'label': f, 'value': f} for f in get_dataset_files()],
                                placeholder=t("Select-dataset",lang),
                            ),
                            ]
                        ),
                        width=6,
                    ),
                    dbc.Col(
                        html.Div(
                            id='upload-div',
                            children=[
                                dcc.Upload(
                                    id="upload-data",
                                    children=html.Div([t("Drag-drop",lang)]),
                                    style={
                                        "width": "100%",
                                        "height": "60px",
                                        "lineHeight": "60px",
                                        "border" : "#007bff",
                                        "borderWidth": "1px",
                                        "borderStyle": "dashed",
                                        "borderRadius": "5px",
                                        "textAlign": "center",
                                        "margin": "10px",
                                    },
                                    multiple=False,
                                ),
                                html.Br(),
                            ]
                        ),
                        width=6,
                    )
                ],
                align="center",  # Vertically align the components in the center
                className="mb-4",  # Add some bottom margin for spacing
            ),
                # A component that shows a message to the user about the data file.
            dbc.Alert([t("Msg-load-file",g.lang)],
                        id="alert-auto",
                        is_open=False,
                        dismissable=True,
                        fade=True,
                        ),

            html.Div( id = "card",
                children=[
                    dbc.Card(
                        children=[
                            # A card with information about the web aplication.
                            dbc.CardHeader(t("Welcome", lang)),
                            dbc.CardBody(
                                [
                                    dcc.Markdown(
                                        t("welcome-card",lang),
                                        style={"margin": "0 10px"},
                                    )
                                ]
                            ),
                        ]
                    ),
                    html.Br(),
                ],                
            ),

            html.Div(id="page-content"),
            html.Div(id="error-message"),
            html.Div(id="table-div"),  
            
            dbc.Row(
                [
                    dbc.Col(
                        [   
                            html.Br(),
                            html.Button(t("Download-CSV", lang), id="download-csv-index", n_clicks=0, hidden = True),
                        ],
                        width=12,
                        className="text-center",
                    ),
                ],
            ),
            ]
    ), lang


# Update this callback to handle file selection from dropdown
@app.callback(
    [Output('stored-data2', 'data'), Output('error-message', 'children')],
    [Input('upload-data', 'contents'),
     Input('upload-data', 'filename'),
     Input('dataset-dropdown', 'value'),
     Input('language', 'data')]
)
def update_data(contents, filename, dataset_value, lang):
    """
    It takes the contents of the uploaded file or a selected file from datasets folder and converts it to a dataframe.
    The dataframe is then converted to a dictionary to be returned.
    """
    if contents:
        # When the user uploads a file
        content_type, content_string = contents.split(",")

        decoded = base64.b64decode(content_string)
        try:
            if filename.endswith(".csv"):
                # Assume that the user uploaded a CSV file
                df = pd.read_csv(io.StringIO(decoded.decode("utf-8")), header=0)
            elif filename.endswith(".arff"):
                # Assume that the user uploaded an ARFF file
                data, meta = arff.loadarff(io.StringIO(decoded.decode("utf-8")))
                df = pd.DataFrame(data)
            elif filename.endswith(".xlsx") or filename.endswith(".xls"):
                df = pd.read_excel(io.BytesIO(decoded),header=0)
            else:
                return None, dbc.Alert(t("Unsoported-file",lang), color="danger")

        except Exception as e:
            print(e)
            return None, dbc.Alert(f"Error: {str(e)}", color="danger")

    elif dataset_value:
        # When the user selects a file from the dropdown
        dataset_path = os.path.join('dataset', dataset_value)
        try:
            if dataset_value.endswith(".csv"):
                # Assume that the user selected a CSV file
                df = pd.read_csv(dataset_path, header=0)
            elif dataset_value.endswith(".arff"):
                # Assume that the user selected an ARFF file
                data, meta = arff.loadarff(dataset_path)
                df = pd.DataFrame(data)
            elif dataset_value.endswith(".xlsx") or dataset_value.endswith(".xls"):
                df = pd.read_excel(dataset_path, header=0)
            else:
                return None, dbc.Alert(t("Unsoported-file",lang), color="danger")

        except Exception as e:
            print(e)
            return None, dbc.Alert(f"Error: {str(e)}", color="danger")
    else:
        return None, None

    # Ensure the DataFrame is JSON serializable
    data_records = df.to_dict('records')
    for record in data_records:
        for key, value in record.items():
            if isinstance(value, bytes):
                record[key] = value.decode('utf-8')

    return data_records, None


@app.callback(
    Output('table-div', 'children'),
    Output('card', 'hidden'),
    [Input('stored-data2', 'data')]
)
def display_table(data):
    """
    Display the data in a table format.

    :param data: The data from the uploaded file
    :return: A DataTable component
    """
    if data:
        df = pd.DataFrame(data)
        return dash_table.DataTable(
            id='editable-table',
            data=df.to_dict('records'),
            columns=[{"name": i, "id": i} for i in df.columns],
            page_size=10,
            editable=True, # Hacer la tabla editable
            row_deletable=True,
        ), True
    return None, False

@app.callback(
    [Output('stored-data', 'data'),
     Output("download-csv-index", "hidden", allow_duplicate= True)],
    Input('editable-table', 'data'),
    State('editable-table', 'data_previous'),
    prevent_initial_call='initial_duplicate'
)
def update_edited_data(rows, previous_rows):
    """
    Captura los datos editados de la tabla.

    :param rows: Datos actuales de la tabla
    :param previous_rows: Datos anteriores de la tabla
    :return: Datos editados para almacenar
    """
    if previous_rows is None:
        raise dash.exceptions.PreventUpdate
    return rows, False

@app.callback(
    Output('stored-data2', 'data', allow_duplicate=True),
    Input('stored-data', 'data'),
    prevent_initial_call='initial_duplicate'
)
def store_edited_data(edited_data):
    """
    Almacena los datos editados.

    :param edited_data: Datos editados de la tabla
    :return: Datos para almacenar
    """
    if edited_data is None:
        raise dash.exceptions.PreventUpdate
    return edited_data

# Callback para descargar los datos editados en un CSV
@app.callback(
    Output("download-component-index", "data"),
    Input("download-csv-index", "n_clicks"),
    Input('upload-data', 'filename'),
    Input('dataset-dropdown', 'value'),
    State("stored-data2", "data"),
)
def download_csv(n_clicks, filename, filename1, rows):
    if rows is None:
        return None
    if n_clicks>0:
    # Convertir los datos editados en un DataFrame
        df_updated = pd.DataFrame(rows)
        
        # Convertir el DataFrame a CSV
        csv_string = df_updated.to_csv(index=False, encoding='utf-8')
        
        if filename is None:
            # Retornar un objeto que permita descargar el archivo CSV
            name = os.path.splitext(filename1)[0] + '.csv'
            return dcc.send_string(csv_string, name)
        else:
            name = os.path.splitext(filename1)[0] + '.csv'
            return dcc.send_string(csv_string, name)


@app.callback(Output("page-content", "children"), Output('upload-div', 'hidden'), Output("alert-auto", "is_open"), Output('table-div', 'hidden'), Output('dropdown-div', 'hidden'), Output('download-csv-index', 'hidden', allow_duplicate= True), Output('language-row','hidden'),
              [Input("url", "pathname"), Input('stored-data2', 'data'), Input('language', 'data')],
              prevent_initial_call=True
)
def display_page_content(pathname, data, lang):
    """
    If the path is empty, return the home page. If the path is "dash", return the dashboard page. If the
    path is "train", return the models page. If the path is "predict", return the predict page.
    Otherwise, return a 404 page

    :param pathname: The pathname argument is the current location of the page
    :param data: The dataframe that is uploaded by the user
    :return: a list of dash components.
    """
    path = app.strip_relative_path(pathname)
    if not path:
        if data != None:
            return pages.home.layout(), False, False, False, False, False, False
        else:
            return pages.home.layout(), False, True, False, False, True, False
    elif path == "dash":
        if data != None:
            return pages.dashboard.layout(lang), True, False, True, True, True, True
        else:
            return [dbc.Modal(  # A modal that is shown when the user tries to access the page without having uploaded a data file.
                [
                    dbc.ModalHeader(dbc.ModalTitle("ERROR"), close_button=False),
                    dbc.ModalBody(
                        [html.I(className="bi bi-exclamation-circle fa-2x"), t("No-data", lang)]),
                    dbc.ModalFooter(dbc.Button([dcc.Link(t('Go-back', lang), href='/', style={'color': 'white'}), ])),
                ],
                id="modal-fs",
                is_open=True,
                keyboard=False,
                backdrop="static",
            ), ], False, False, False, False, False, False
    elif path == "metafeatures":
        if data != None:
            return pages.metafeatures.layout(lang), True, False, True, True, True, True
        else:
            return [dbc.Modal(  # A modal that is shown when the user tries to access the page without having uploaded a data file.
                [
                    dbc.ModalHeader(dbc.ModalTitle("ERROR"), close_button=False),
                    dbc.ModalBody(
                        [html.I(className="bi bi-exclamation-circle fa-2x"), t("No-data", lang)]),
                    dbc.ModalFooter(dbc.Button([dcc.Link(t('Go-back', lang), href='/', style={'color': 'white'}), ])),
                ],
                id="modal-fs",
                is_open=True,
                keyboard=False,
                backdrop="static",
            ), ], False, False, False, False, False
    elif path == "complexity":
        # Checking if exists a model in the server. If it does, it returns the predict page.
        if data != None:
            return pages.complexity.layout(lang), True, False, True, True, True, True
        else:
            return [dbc.Modal(  # A modal that is shown when the user tries to access the page without having uploaded a data file.
                [
                    dbc.ModalHeader(dbc.ModalTitle("ERROR"), close_button=False),
                    dbc.ModalBody(
                        [html.I(className="bi bi-exclamation-circle fa-2x"), t("No-data", lang)]),
                    dbc.ModalFooter(dbc.Button([dcc.Link(t('Go-back', lang), href='/', style={'color': 'white'}), ])),
                ],
                id="modal-fs",
                is_open=True,
                keyboard=False,
                backdrop="static",
            ), ], False, False, False, False, False, False
    else:
        return "404"

#Dashboard Callbacks

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

#Complexity Callbacks

@app.callback(
    Output('download-component_', 'data'),
    Input("download-csv_", "n_clicks"),
    State("stored-data2", "data"),
    State("input-feature-dropdown_", "value"),
    State("class-feature-dropdown_", "value"),
)
def update_download_link(n_clicks, data, input_feature, class_feature):
    if n_clicks > 0 and data and input_feature and class_feature:
        df = pd.DataFrame(data)   
        # Initialize ComplexityCalculator with default parametrization
        cc = px.ComplexityCalculator()

        X , y = df[input_feature], df[class_feature]

        # Convert categorical columns to numerical representations
        for column in X.columns:
            if X[column].dtype == 'object':  # Check if column has string (object) values
                X[column] = pd.factorize(X[column])[0] 

        # Fit model with data
        cc.fit(X, y)
        
        cc._metrics()
        
        # Get the report dictionary
        report = cc.report()

        complexities = report.pop('complexities')

        # Combinar los dos diccionarios
        report.update(complexities)

        # Convertir a DataFrame
        results_df = pd.DataFrame([report])
        
        # Crear el enlace de descarga
        return dcc.send_data_frame(results_df.to_csv, "complexity.csv")

@app.callback(
    Output("input-feature-dropdown_", "options"),
    Output("class-feature-dropdown_", "options"),
    Input("stored-data2", "data"),
)
def update_dropdown_options(data):
    if data:
        df = pd.DataFrame(data)
        options = [{"label": col, "value": col} for col in df.columns]
        return options, options
    return [], []

@app.callback(
    [Output("output-image", "src"),
     Output("output-image", "hidden"),
     Output("download-csv_", "hidden"),
     Output("complexity-report", "children"),
     Output("error-message_", "children")],
    Input("calculate-measures", "n_clicks"),
    State('language', 'data'),
    State("stored-data2", "data"),
    State("input-feature-dropdown_", "value"),
    State("class-feature-dropdown_", "value"),
)
def calculate_complexity_measures(n_clicks, lang, data, input_feature, class_feature):
    if n_clicks > 0:
        if not input_feature or not class_feature:
            return '', True, True, '', dbc.Alert(t("Plot-Alert", lang), color="danger")

        df = pd.DataFrame(data)
        # Initialize ComplexityCalculator with default parametrization
        cc = px.ComplexityCalculator()

        X, y = df[input_feature], df[class_feature]

        # Convert categorical columns to numerical representations
        for column in X.columns:
            if X[column].dtype == 'object':  # Check if column has string (object) values
                X[column] = pd.factorize(X[column])[0]

        # Fit model with data
        cc.fit(X, y)

        cc._metrics()

        # Prepare figure
        fig = plt.figure(figsize=(7, 7))

        # Generate plot describing the dataset
        cc.plot(fig, (1, 1, 1))

        # Save the figure in a buffer in memory
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close(fig)

        # Get the report dictionary
        report = cc.report()

        # Convert the report dictionary into Dash HTML components with tooltips
        tables = []

        # Table for linearity measures
        l_metrics = [("l1", report['complexities']['l1']),
                    ("l2", report['complexities']['l2']),
                    ("l3", report['complexities']['l3'])]
        l_table = [html.Tr([html.Th(t("Metric", lang)), html.Th(t("Value", lang))])]
        for key, value in l_metrics:
            description = t(key,lang)
            l_table.append(html.Tr([html.Td(key, title=description), html.Td(value)]))
        tables.append(html.Div([html.H3(t("Linearity-measures", lang)),html.Hr(), html.Table(l_table), html.Br()]))

        # Table for feature based measures
        f_metrics = [("f1", report['complexities']['f1']),
                    ("f1v", report['complexities']['f1v']),
                    ("f2", report['complexities']['f2']),
                    ("f3", report['complexities']['f3']),
                    ("f4", report['complexities']['f4'])]
        f_table = [html.Tr([html.Th(t("Metric", lang)), html.Th(t("Value", lang))])]
        for key, value in f_metrics:
            description = t(key,lang)
            f_table.append(html.Tr([html.Td(key, title=description), html.Td(value)]))
        tables.append(html.Div([html.H3(t("Feature-based-measures", lang)),html.Hr(), html.Table(f_table), html.Br()]))

        # Table for class imbalance measures
        c_metrics = [("c1", report['complexities']['c1']),
                    ("c2", report['complexities']['c2'])]
        c_table = [html.Tr([html.Th(t("Metric", lang)), html.Th(t("Value", lang))])]
        for key, value in c_metrics:
            description = t(key,lang)
            c_table.append(html.Tr([html.Td(key, title=description), html.Td(value)]))
        tables.append(html.Div([html.H3(t("Class-imbalance-measures", lang)),html.Hr(), html.Table(c_table), html.Br()]))

        # Table for dimensionality measures
        t_metrics = [("t2", report['complexities']['t2']),
                    ("t3", report['complexities']['t3']),
                    ("t4", report['complexities']['t4'])]
        t_table = [html.Tr([html.Th(t("Metric", lang)), html.Th(t("Value", lang))])]
        for key, value in t_metrics:
            description = t(key,lang)
            t_table.append(html.Tr([html.Td(key, title=description), html.Td(value)]))
        tables.append(html.Div([html.H3(t("Dimensionality-measures", lang)),html.Hr(), html.Table(t_table), html.Br()]))

        # Table for network measures
        h_metrics = [("hubs", report['complexities']['hubs']),
                    ("clsCoef", report['complexities']['clsCoef']),
                    ("density", report['complexities']['density'])]
        h_table = [html.Tr([html.Th(t("Metric", lang)), html.Th(t("Value", lang))])]
        for key, value in h_metrics:
            description = t(key,lang)
            h_table.append(html.Tr([html.Td(key, title=description), html.Td(value)]))
        tables.append(html.Div([html.H3(t("Network-measures", lang)),html.Hr(), html.Table(h_table), html.Br()]))

        # Table for neighborhood measures
        ln_metrics = [("lsc", report['complexities']['lsc']),
                    ("t1", report['complexities']['t1']),
                    ("n1", report['complexities']['n1']),
                    ("n2", report['complexities']['n2']),
                    ("n3", report['complexities']['n3']),
                    ("n4", report['complexities']['n4'])]
        ln_table = [html.Tr([html.Th(t("Metric", lang)), html.Th(t("Value", lang))])]
        for key, value in ln_metrics:
            description = t(key,lang)
            ln_table.append(html.Tr([html.Td(key, title=description), html.Td(value)]))
        tables.append(html.Div([html.H3(t("Neighborhood-measures", lang)),html.Hr(), html.Table(ln_table),html.Br()]))

        # Render all tables together
        final_report = html.Div(tables)

        # Return the image, the hidden state, and the table
        return f'data:image/png;base64,{image_base64}', False, False, final_report, None
    
    return '', True, True, '', None  # Return empty values by default


#Metafeatures callbacks

@app.callback(
    Output('download-component', 'data'),
    Input("download-csv", "n_clicks"),
    State("stored-data2", "data"),
    State("input-feature-dropdown1", "value"),
    State("class-feature-dropdown1", "value"),
)
def update_download_link(n_clicks, data, input_feature, class_feature):
    if n_clicks > 0 and data and input_feature and class_feature:
        df = pd.DataFrame(data)
        
        # Ensure the input features are in the correct format
        X = df[input_feature].values
        y = df[class_feature].values

        # Reshape if X is a single feature
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Extraer las meta-características
        mfe = MFE(groups=["general", "statistical", "info-theory"])
        mfe.fit(X, y)
        ft_names, ft_values = mfe.extract()
        
        # Crear un DataFrame con los resultados
        results_df = pd.DataFrame({'Feature': ft_names, 'Value': ft_values})
        
        # Crear el enlace de descarga
        return dcc.send_data_frame(results_df.to_csv, "metafeatures.csv")

@app.callback(
    Output("input-feature-dropdown1", "options"),
    Output("class-feature-dropdown1", "options"),
    Input("stored-data2", "data"),
)
def update_dropdown_options_(data):
    if data:
        df = pd.DataFrame(data)
        options = [{"label": col, "value": col} for col in df.columns]
        return options, options
    return [], []

@app.callback(
    [Output("output-image", "src"),
     Output("output-image", "hidden"),
     Output("download-csv_", "hidden"),
     Output("output-meta-features", "children"),
     Output("error-message_", "children"),
     Output("error-message1", "children")],
     Input("calculate_meta_features", "n_clicks"),
    State('language', 'data'),
    State("stored-data2", "data"),
    State("input-feature-dropdown1", "value"),
    State("class-feature-dropdown1", "value"),
)
def calculate_measures_and_meta_features(n_clicks, lang, data, input_feature, class_feature):
    # Initialize outputs
    image_output = ''
    image_hidden = True
    download_hidden = True
    meta_features_output = html.Div()
    error_message = None
    error_message1 = None

    # Handle complexity measures calculation
    if n_clicks > 0:
        if not input_feature or not class_feature:
            error_message = dbc.Alert(t("Plot-Alert", lang), color="danger")
            return image_output, image_hidden, download_hidden, meta_features_output, error_message, None

        df = pd.DataFrame(data)
        cc = px.ComplexityCalculator()
        X, y = df[input_feature], df[class_feature]

        # Convert categorical columns to numerical representations
        for column in X.columns:
            if X[column].dtype == 'object':
                X[column] = pd.factorize(X[column])[0]

        cc.fit(X, y)
        cc._metrics()
        
        # Prepare figure
        fig = plt.figure(figsize=(7, 7))
        cc.plot(fig, (1, 1, 1))
        
        # Save the figure in a buffer
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close(fig)

        # Generate the report
        report = cc.report()
        tables = []
        # (Table generation code omitted for brevity, use the original logic to create tables)
        final_report = html.Div(tables)

        image_output = f'data:image/png;base64,{image_base64}'
        image_hidden = False
        download_hidden = False

        return image_output, image_hidden, download_hidden, final_report, None, error_message1

    # Handle meta features calculation
    if n_clicks > 0:
        if not input_feature or not class_feature:
            error_message1 = dbc.Alert(t("Plot-Alert", lang), color="danger")
            return image_output, image_hidden, download_hidden, meta_features_output, error_message, error_message1

        if data:
            df = pd.DataFrame(data)
            X = df[input_feature].values
            y = df[class_feature].values

            if X.ndim == 1:
                X = X.reshape(-1, 1)

            groups = ["clustering", "concept", "general", "statistical", "info-theory"]
            tables = []

            for group in groups:
                mfe = MFE(groups=[group])
                mfe.fit(X, y)
                ft_names, ft_values = mfe.extract()

                # Create rows for the table
                table_rows = []
                for feature_name, feature_value in zip(ft_names, ft_values):
                    description = t(feature_name, lang)
                    table_rows.append(
                        html.Tr([
                            html.Td(feature_name, title=description),
                            html.Td(feature_value)
                        ])
                    )

                if table_rows:
                    tables.append(html.Div([
                        html.H5(f"{t('Meta-Features-of-the-group', lang)}: {group.capitalize()}"),
                        dbc.Table([
                            html.Thead(html.Tr([html.Th(t("Feature", lang)), html.Th(t("Value", lang))])),
                            html.Tbody(table_rows)
                        ], striped=True, bordered=True, hover=True)
                    ]))

            meta_features_output = html.Div(tables)

            return image_output, image_hidden, download_hidden, meta_features_output, error_message, error_message1

    return image_output, True, True, meta_features_output, None, None  # Default return values



# A way to run the app in a local server.
if __name__ == "__main__":
    app.run_server(debug=True)

