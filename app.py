# app.py (Dashboard con rango de años y selección múltiple de países)
import os
import joblib
import pandas as pd
import numpy as np
import dash
from dash import dcc, html, Input, Output
import plotly.express as px
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

# --------------------------
# Cargar datos y modelos
# --------------------------
df = pd.read_csv("Runups.csv", low_memory=False)
df = df.dropna(subset=['MAXIMUM_HEIGHT','DISTANCE_FROM_SOURCE','TRAVEL_TIME_HOURS','PERIOD'])

rf_reg_path = "rf_regressor.pkl"
rf_clf_path = "rf_classifier.pkl"

rf_reg = joblib.load(rf_reg_path) if os.path.exists(rf_reg_path) else None
rf_clf = joblib.load(rf_clf_path) if os.path.exists(rf_clf_path) else None

# --------------------------
# App
# --------------------------
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

# --------------------------
# Layout
# --------------------------
app.layout = html.Div(
    style={'backgroundColor': '#f9f9f9', 'padding': '20px'},
    children=[
        html.H1("Dashboard: Análisis de Tsunamis - Altura de Olas",
                style={'textAlign': 'center', 'color': '#003366'}),
        html.H3("Exploración, Regresión y Clasificación de Olas",
                style={'textAlign': 'center', 'color': '#006699'}),

        # filtros: rango de años y percentil
        html.Div([
            html.Div([
                html.Label("Filtrar por rango de años", style={'fontWeight': 'bold'}),
                dcc.RangeSlider(
                    id="year-slider",
                    min=int(df['YEAR'].min()), 
                    max=int(df['YEAR'].max()),
                    value=[int(df['YEAR'].min()), int(df['YEAR'].max())],
                    marks={y: str(y) for y in range(int(df['YEAR'].min()), int(df['YEAR'].max())+1, 5)},
                    step=1
                )
            ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px'}),

            html.Div([
                html.Label("Selecciona umbral para 'ola alta' (percentil)", style={'fontWeight': 'bold'}),
                dcc.Slider(
                    id='percentil-umbral', min=50, max=99, step=1, value=90,
                    marks={50: '50', 75: '75', 90: '90', 95: '95', 99: '99'}
                )
            ], style={'width': '48%', 'display': 'inline-block', 'float': 'right', 'padding': '10px'}),
        ], style={'marginBottom': '20px'}),

        # filtro por países (multi)
        html.Div([
            html.Label("Filtrar por país", style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id="country-dropdown",
                options=[{"label": c, "value": c} for c in sorted(df['COUNTRY'].dropna().unique())],
                value=[],
                multi=True,
                placeholder="Selecciona uno o más países"
            ),
            html.Button("Reset filtros", id="reset-btn", n_clicks=0, style={'marginTop': '6px'})
        ], style={'marginBottom': '20px'}),

        # tabs con gráficas
        dcc.Tabs([
            dcc.Tab(label='Exploración', children=[
                dcc.Graph(id='scatter-mag-alt', style={'backgroundColor': 'white', 'padding': '10px'}),
                dcc.Graph(id='hist-altura', style={'backgroundColor': 'white', 'padding': '10px'})
            ]),

            dcc.Tab(label='Modelo - Regresión', children=[
                html.Div(id='reg-metrics', style={'marginTop': '20px'}),
                dcc.Graph(id='pred-vs-true', style={'backgroundColor': 'white', 'padding': '10px'})
            ]),

            dcc.Tab(label='Modelo - Clasificación', children=[
                html.Div(id='clf-metrics', style={'marginTop': '20px'}),
                dcc.Graph(id='roc-curve'),
                html.P("La curva ROC muestra el desempeño del modelo. Un AUC cercano a 0.65 "
                       "indica que el modelo tiene un poder predictivo moderado."),
                dcc.Graph(id='conf-matrix'),
                html.P("La matriz de confusión evidencia que el modelo detecta mejor olas bajas (clase 0) "
                       "que olas altas (clase 1). Esto ocurre porque los eventos extremos son poco frecuentes.")
            ])
        ])
    ]
)

# --------------------------
# Callbacks
# --------------------------

# Reset países
@app.callback(
    Output("country-dropdown", "value"),
    Input("reset-btn", "n_clicks")
)
def reset_country(n_clicks):
    if n_clicks and n_clicks > 0:
        return []
    return dash.no_update

# Scatter: rango de años + multi país
@app.callback(
    Output('scatter-mag-alt', 'figure'),
    Input('year-slider', 'value'),
    Input('country-dropdown', 'value')
)
def update_scatter(year_range, countries):
    y0, y1 = year_range
    dff = df[(df['YEAR'] >= y0) & (df['YEAR'] <= y1)]

    if countries and len(countries) > 0:
        dff = dff[dff['COUNTRY'].isin(countries)]

    try:
        if os.path.exists("Eventos.csv"):
            evt = pd.read_csv("Eventos.csv", low_memory=False)
            if 'SOURCE_ID' in evt.columns and 'SOURCE_ID' in dff.columns:
                merged = pd.merge(evt[['SOURCE_ID','PRIMARY_MAGNITUDE']], dff, on='SOURCE_ID', how='inner')
                fig = px.scatter(merged, x='PRIMARY_MAGNITUDE', y='MAXIMUM_HEIGHT',
                                 hover_data=['COUNTRY','YEAR'], color='PRIMARY_MAGNITUDE',
                                 title=f"Magnitud vs Altura ({y0}-{y1})")
            else:
                fig = px.scatter(dff, x='DISTANCE_FROM_SOURCE', y='MAXIMUM_HEIGHT',
                                 hover_data=['COUNTRY','YEAR'], color='DISTANCE_FROM_SOURCE',
                                 title=f"Distancia vs Altura ({y0}-{y1})")
        else:
            fig = px.scatter(dff, x='DISTANCE_FROM_SOURCE', y='MAXIMUM_HEIGHT',
                             hover_data=['COUNTRY','YEAR'], color='DISTANCE_FROM_SOURCE',
                             title=f"Distancia vs Altura ({y0}-{y1})")
    except Exception as e:
        fig = px.scatter(pd.DataFrame({'x':[0],'y':[0]}), x='x', y='y',
                         title=f"No hay datos (error: {str(e)})")

    fig.update_layout(plot_bgcolor='#f0f0f0')
    return fig

# Histograma: rango + multi país
@app.callback(
    Output('hist-altura','figure'),
    Input('year-slider','value'),
    Input('country-dropdown','value')
)
def update_hist(year_range, countries):
    y0, y1 = year_range
    dff = df[(df['YEAR'] >= y0) & (df['YEAR'] <= y1)]

    if countries and len(countries) > 0:
        dff = dff[dff['COUNTRY'].isin(countries)]

    fig = px.histogram(dff, x='MAXIMUM_HEIGHT', nbins=100,
                       title=f"Histograma de alturas ({y0}-{y1})",
                       color_discrete_sequence=['#003366'])
    fig.update_layout(plot_bgcolor='#f0f0f0')
    return fig

# Métricas de regresión
@app.callback(
    Output('reg-metrics','children'),
    Input('year-slider','value')
)
def show_reg_metrics(year_range):
    if os.path.exists("regression_metrics.csv"):
        rm = pd.read_csv("regression_metrics.csv")
        table = html.Table(
            [html.Tr([html.Th(c) for c in rm.columns], style={'backgroundColor':'#003366','color':'white'})] +
            [html.Tr([html.Td(rm.iloc[i][c]) for c in rm.columns]) for i in range(len(rm))],
            style={'border':'1px solid black', 'marginTop':'10px'}
        )
        return html.Div([html.H4("Métricas de regresión (archivo)", style={'color':'#003366'}), table])
    return "No hay métricas de regresión generadas. Se corre analisis_modelado.py primero."

# Predicciones vs verdadero
@app.callback(
    Output('pred-vs-true','figure'),
    Input('year-slider','value')
)
def pred_vs_true(year_range):
    if os.path.exists("predicciones_test_clasificacion.csv") and rf_reg is not None:
        dff = pd.read_csv("predicciones_test_clasificacion.csv")
        fig = px.scatter(dff, x='prob_high90', y='true_high90', hover_data=['pred_high90'],
                         color='prob_high90', color_continuous_scale='Blues',
                         title="Probabilidad predicha vs verdadero (clasificación)")
        fig.update_layout(plot_bgcolor='#f0f0f0')
        return fig
    return px.scatter(pd.DataFrame({'x':[0],'y':[0]}), x='x', y='y', title="Ejecución analisis_modelado.py para generar predicciones")

# Métricas de clasificación
@app.callback(
    Output('clf-metrics','children'),
    Input('percentil-umbral','value')
)
def update_clf_metrics(percentil):
    if os.path.exists("classification_metrics.csv"):
        cm = pd.read_csv("classification_metrics.csv")
        table = html.Table(
            [html.Tr([html.Th(c) for c in cm.columns], style={'backgroundColor':'#006699','color':'white'})] +
            [html.Tr([html.Td(cm.iloc[i][c]) for c in cm.columns]) for i in range(len(cm))],
            style={'border':'1px solid black', 'marginTop':'10px'}
        )
        return html.Div([html.H4(f"Métricas de clasificación (umbral {percentil} percentil)", style={'color':'#006699'}), table])
    return "Se corre analisis_modelado.py primero para generar métricas de clasificación."

# ROC y matriz de confusión
@app.callback(
    Output('roc-curve','figure'),
    Output('conf-matrix','figure'),
    Input('percentil-umbral','value')
)
def update_roc_cm(percentil):
    if os.path.exists("predicciones_test_clasificacion.csv"):
        dff = pd.read_csv("predicciones_test_clasificacion.csv")
        try:
            fpr, tpr, thr = roc_curve(dff['true_high90'], dff['prob_high90'])
            roc_fig = px.area(x=fpr, y=tpr,
                              title=f"ROC curve (AUC={roc_auc_score(dff['true_high90'], dff['prob_high90']):.3f})",
                              color_discrete_sequence=['#006699'])
            roc_fig.update_xaxes(title="False Positive Rate")
            roc_fig.update_yaxes(title="True Positive Rate")
        except Exception:
            roc_fig = px.scatter(title="No se pudo calcular ROC")

        thresh = np.percentile(dff['prob_high90'], percentil)
        preds = (dff['prob_high90'] >= thresh).astype(int)
        cm = confusion_matrix(dff['true_high90'], preds)
        cm_df = pd.DataFrame(cm, index=['true_0', 'true_1'], columns=['pred_0', 'pred_1'])
        cm_fig = px.imshow(cm_df, text_auto=True,
                           title=f"Confusión (umbral percentil {percentil} -> prob >= {thresh:.3f})",
                           color_continuous_scale='Blues')
        return roc_fig, cm_fig

    return px.scatter(title="Ejecución analisis_modelado.py primero"), px.imshow([[0, 0], [0, 0]], title="Sin datos")

# --------------------------
# Run server
# --------------------------
if __name__ == '__main__':
    app.run(debug=True)
