# -*- coding: utf-8 -*-
"""
| Main for visual interface
| Author: Alexandre CARRE (alexandre.carre@gustaveroussy.fr)
| Created on: Jan 14, 2021
"""
import base64
import io

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import waitress
from ComScan.utils import tsne, u_map, column_var_dtype
from dash.dependencies import Input, Output
from dash_table.Format import Format, Scheme
from sklearn.preprocessing import LabelEncoder

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.config["suppress_callback_exceptions"] = True
server = app.server

colors = {
    "graphBackground": "#F5F5F5",
    "background": "#ffffff",
    "text": "#000000"
}

app.layout = html.Div([
    # drag & drop
    dcc.Upload(
        id="upload-data",
        children=html.Div([
            "Drag and Drop or ",
            html.A("Select Files")
        ]),
        style={
            "width": "100%",
            "height": "60px",
            "lineHeight": "60px",
            "borderWidth": "1px",
            "borderStyle": "dashed",
            "borderRadius": "5px",
            "textAlign": "center",
            "margin": "10px"
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),

    html.Div(id="get-fn"),

    html.Div([
        dash_table.DataTable(
            id="table-filtering-be",
            filter_action="custom",
            filter_query="",
            style_table={"overflowX": "scroll",
                         "height": "300px",
                         "overflowY": "auto",
                         "textOverflow": "hidden"
                         },
            editable=True,
            css=[
                {"selector": ".column-header--delete svg", "rule": 'display: "none"'},
                {"selector": ".column-header--delete::before", "rule": 'content: "X"'}
            ],
            merge_duplicate_headers=True,
        ),
    ]),

    # load file
    # html.Div(id='output-data-upload'),

    # allow to only display when a drag & drop is done
    html.Div(id="display-all-plot", children=[
        html.Hr(),
        html.Div([
            html.Div([
                html.Div([
                    html.Button("Parallel Coordinate", id="btn-nclicks-pc", n_clicks=0),  # Button pc
                    html.Button("Bar Chart", id="btn-nclicks-bc", n_clicks=0),  # Button bc
                ]),
                dcc.Dropdown(id="dropdown-menu-pcbc", multi=False, placeholder="Select scale"),
            ], style={"width": "345px"}),
            html.Div(id="button-state-pcbc", style={"display": "none"}),  # hidden state to keep state of the button
            dcc.Graph(id="figure-pcbc"),
        ]),

        html.Hr(),

        html.Div([
            html.Div([
                html.H4("Scatter Plot"),
                html.Div([
                    dcc.Dropdown(id="dropdown-menu-scatter-y",
                                 multi=False,
                                 placeholder="Select y",
                                 style={"width": "100%"},
                                 ),
                    dcc.Dropdown(id="dropdown-menu-scatter-x",
                                 multi=False,
                                 placeholder="Select x",
                                 style={"width": "100%"},

                                 ),
                ], style={"display": "flex"},
                ),
                dcc.Graph(id="figure-scatter-matrix", )
            ], className="six columns"),

            html.Div([
                html.H4("Pearson correlation matrix"),
                dcc.Dropdown(id="dropdown-menu-heatmap",
                             placeholder="Select features",
                             multi=True,
                             ),
                dcc.RadioItems(id="radio-heatmap",
                               options=[
                                   {"label": "Drop NaNs", "value": "drop"},
                                   {"label": "Initial State", "value": "init"},
                               ],
                               value="init",
                               labelStyle={"display": "inline-block"}
                               ),
                dcc.Graph(id="figure-heatmap", style={"margin": "auto"})
            ], className="six columns"),

        ], className="row"),

        html.Hr(),

        html.Div([
            html.Div([
                html.H4("t-SNE"),
                html.Div([
                    dcc.RadioItems(id="radio-tsne",
                                   options=[
                                       {"label": "2-D", "value": 2},
                                       {"label": "3-D", "value": 3},
                                   ],
                                   value=2,
                                   labelStyle={"display": "inline-block"}
                                   ),
                ], style={"display": "flex"},
                ),
                dcc.Graph(id="figure-tsne"),
                html.Div(id="figure-tsne-sc", style={"whiteSpace": "pre-line"})
            ], className="six columns"),

            html.Div([
                html.H4("UMAP"),
                html.Div([
                    dcc.RadioItems(id="radio-umap",
                                   options=[
                                       {"label": "2-D", "value": 2},
                                       {"label": "3-D", "value": 3},
                                   ],
                                   value=2,
                                   labelStyle={"display": "inline-block"}
                                   ),
                ], style={"display": "flex"},
                ),
                dcc.Graph(id="figure-umap"),
                html.Div(id="figure-umap-sc", style={"whiteSpace": "pre-line"})
            ], className="six columns"),

        ], className="row"),
    ],
             style={"display": "none"})
])


@app.callback(
    Output("display-all-plot", "style"),
    [Input("table-filtering-be", "columns")])
def display_all_plot(df):
    if df:
        return {"display": "block"}
    else:
        return {"display": "none"}


@app.callback(
    Output("get-fn", "children"),
    [
        Input("upload-data", "contents"),
        Input("upload-data", "filename"),
    ])
def get_df(contents, filename):
    fn = html.Div()
    if contents:
        filename = filename[0]
        fn = html.Div([
            html.H5(filename)])
    return fn


def parse_data(contents, filename):
    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    df = pd.DataFrame()
    try:
        if "csv" in filename:
            # Assume that the user uploaded a CSV or TXT file
            df = pd.read_csv(
                io.StringIO(decoded.decode("utf-8")))
        elif "xls" in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
        elif "txt" or "tsv" in filename:
            # Assume that the user upl, delimiter = r"\s+"oaded an excel file
            df = pd.read_csv(
                io.StringIO(decoded.decode("utf-8")), delimiter=r"\s+")
    except Exception as e:
        print(e)
        return html.Div([
            "There was an error processing this file."
        ])

    if "id" not in df:
        return html.Div([
            "id columns is missing in file."
        ])

    return df.sort_values(by=["id"])


operators = [["ge ", ">="],
             ["le ", "<="],
             ["lt ", "<"],
             ["gt ", ">"],
             ["ne ", "!="],
             ["eq ", "="],
             ["contains "],
             ["datestartswith "]]


def split_filter_part(filter_part):
    for operator_type in operators:
        for operator in operator_type:
            if operator in filter_part:
                name_part, value_part = filter_part.split(operator, 1)
                name = name_part[name_part.find("{") + 1: name_part.rfind("}")]

                value_part = value_part.strip()
                v0 = value_part[0]
                if v0 == value_part[-1] and v0 in ("'", '"', '`'):
                    value = value_part[1: -1].replace("\\" + v0, v0)
                else:
                    try:
                        value = float(value_part)
                    except ValueError:
                        value = value_part

                # word operators need spaces after them in the filter string,
                # but we don"t want these later
                return name, operator_type[0].strip(), value

    return [None] * 3


@app.callback(
    [Output("table-filtering-be", "data"),
     Output("table-filtering-be", "columns"),
     Output("table-filtering-be", "export_format"),
     ],
    [Input("upload-data", "contents"),
     Input("upload-data", "filename"),
     Input("table-filtering-be", "filter_query")])
def update_table(contents, filename, filter_):
    filtering_expressions = filter_.split(" && ")

    df = pd.DataFrame([])
    columns = []
    export_format = None
    if contents:
        contents = contents[0]
        filename = filename[0]
        df = parse_data(contents, filename)

        for filter_part in filtering_expressions:
            col_name, operator, filter_value = split_filter_part(filter_part)

            if operator in ("eq", "ne", "lt", "le", "gt", "ge"):
                # these operators match pandas series operator method names
                df = df.loc[getattr(df[col_name], operator)(filter_value)]
            elif operator == "contains":
                df = df.loc[df[col_name].str.contains(filter_value)]
            elif operator == "datestartswith":
                # this is a simplification of the front-end filtering logic,
                # only works with complete fields in standard format
                df = df.loc[df[col_name].str.startswith(filter_value)]

        columns = [{"name": i,
                    "id": i,
                    "renamable": True, "deletable": [True, False],
                    "format": Format().scheme(Scheme.fixed).precision(2),
                    "type": "numeric",
                    } for i in df.columns]
        export_format = "csv"

    return df.to_dict("records"), columns, export_format


@app.callback(
    Output("hidden-state-df", "children"),
    Output("hidden-state-df-dropnans", "children"),
    Input("table-filtering-be", "data"),
    Input("table-filtering-be", "columns"))
def get_df(rows, columns):
    df = pd.DataFrame(rows, columns=[c["name"] for c in columns])
    df_no_nans = df.copy()
    df_no_nans.dropna(axis=1, inplace=True)
    return df, df_no_nans


@app.callback([
    Output("dropdown-menu-pcbc", "options"),
    Output("dropdown-menu-heatmap", "options"),
    Output("dropdown-menu-heatmap", "value"),
    Output("dropdown-menu-scatter-x", "options"),
    Output("dropdown-menu-scatter-y", "options"),
],
    Input("table-filtering-be", "columns"))
def dropdown_menu_labels(columns):
    y_label = [{"label": k["name"], "value": k["name"]} for k in columns if k["name"] != "id"]
    value = [k["name"] for k in columns if k["name"] != "id"]
    return y_label, y_label, value, y_label, y_label


@app.callback(
    Output("figure-pcbc", "figure"),
    Input("button-state-pcbc", "children"),
    Input("table-filtering-be", "data"),
    Input("table-filtering-be", "columns"),
    Input("dropdown-menu-pcbc", "value")
)
def update_parcoords(button_state, rows, columns, drop_menu_value):
    df = pd.DataFrame(rows, columns=[c["name"] for c in columns])
    df = df_label_encode(df)
    if button_state == "pc":

        fig = go.Figure(data=go.Parcoords(
            dimensions=list(
                [{"range": [df[col].min(), df[col].max()], "label": col, "values": df[col].values} for col in
                 df.columns]),
            labelangle=30, line=dict(color=df[drop_menu_value], colorscale="agsunset",
                                     showscale=True) if drop_menu_value is not None else None))
        return fig
    elif button_state == "bc":
        return px.bar(df, x="id", y=drop_menu_value, )


@app.callback(
    Output("button-state-pcbc", "children"),
    Input("btn-nclicks-pc", "n_clicks"),
    Input("btn-nclicks-bc", "n_clicks"),
)
def get_button_state(*args):
    changed_id = [p["prop_id"] for p in dash.callback_context.triggered][0]
    state = "pc"  # default state or (button_pc == 0 and buttton_nc == 0)
    if "btn-nclicks-pc" in changed_id:
        state = "pc"
    elif "btn-nclicks-bc" in changed_id:
        state = "bc"
    return state


@app.callback(
    Output("figure-scatter-matrix", "figure"),
    Input("table-filtering-be", "data"),
    Input("table-filtering-be", "columns"),

    Input("dropdown-menu-scatter-y", "value"),
    Input("dropdown-menu-scatter-x", "value"))
def update_scatter(rows, columns, drop_menu_value_y, drop_menu_value_x):
    df = pd.DataFrame(rows, columns=[c["name"] for c in columns])
    df = df_clean(df)
    df = df_label_encode(df)
    fig = px.scatter(df, x=drop_menu_value_x, y=drop_menu_value_y, template="plotly_white")
    return fig


@app.callback(
    Output("figure-heatmap", "figure"),
    [Input("table-filtering-be", "data"),
     Input("table-filtering-be", "columns"),
     Input("dropdown-menu-heatmap", "value"),
     Input("radio-heatmap", "value"),
     ])
def filter_heatmap(rows, columns, drop_menu_heatmap, drop):
    df = pd.DataFrame(rows, columns=[c["name"] for c in columns])
    df = df_label_encode(df)
    df = df[drop_menu_heatmap]
    corr = df.corr(method="pearson")
    if drop == "drop":
        corr.dropna(axis=0, how="all", inplace=True)
        corr.dropna(axis=1, how="all", inplace=True)

    layout = go.Layout(
        margin=go.layout.Margin(
            l=50,  # left margin
            r=50,  # right margin
            b=50,  # bottom margin
            t=50  # top margin
        ),
        template="plotly_white",
    )
    fig = go.Figure(data=go.Heatmap(z=corr.values,
                                    x=corr.index.values,
                                    y=corr.columns.values),
                    layout=layout)
    return fig


@app.callback(
    [Output("figure-tsne", "figure"),
     Output("figure-tsne-sc", "children")],
    Input("table-filtering-be", "data"),
    Input("table-filtering-be", "columns"),
    Input("radio-tsne", "value"),
)
def update_tsne(rows, columns, n_components):
    df = pd.DataFrame(rows, columns=[c["name"] for c in columns])
    # drop columns with NaNs
    df = df_clean(df)
    df = df_label_encode(df)
    selected_columns = df.columns.tolist()
    message = f"Selected features: {', '.join(selected_columns)}"

    projections = tsne(df, df.columns.tolist(), n_components=n_components)

    fig = px.scatter(projections, x=0, y=1) if n_components == 2 else px.scatter_3d(projections, x=0, y=1, z=2)

    return fig, message


@app.callback(
    [Output("figure-umap", "figure"),
     Output("figure-umap-sc", "children")],
    Input("table-filtering-be", "data"),
    Input("table-filtering-be", "columns"),
    Input("radio-umap", "value"),
)
def update_umap(rows, columns, n_components):
    df = pd.DataFrame(rows, columns=[c["name"] for c in columns])
    # drop columns with NaNs and encode
    df = df_clean(df)
    df = df_label_encode(df)
    selected_columns = df.columns.tolist()
    message = f"Selected features: {', '.join(selected_columns)}"

    projections = u_map(df, df.columns.tolist(), n_components=n_components)

    fig = px.scatter(projections, x=0, y=1) if n_components == 2 else px.scatter_3d(projections, x=0, y=1, z=2)

    return fig, message


def df_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(axis=1)
    df = df.drop(columns=["id"])
    return df


def df_label_encode(df: pd.DataFrame) -> pd.DataFrame:
    le = LabelEncoder()
    obj_list = column_var_dtype(df, identify_dtypes=("object",))["name"].tolist()
    for feat in obj_list:
        df[feat] = le.fit_transform(df[feat].astype(str))
    return df


if __name__ == "__main__":
    waitress.serve(app.server, host="0.0.0.0", port=8050)
    # app.run_server(debug=True, dev_tools_ui=False, dev_tools_props_check=True)
