# Imports
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
from dash import dash_table
import dash_bootstrap_components as dbc
from dash_dangerously_set_inner_html import DangerouslySetInnerHTML

import pandas as pd
import plotly.express as px
import re
import string
import numpy as np
import nltk

# Pre-processing functions

def text_lowercase(text):
    return text.lower()

def preprocessing(text):
    return text_lowercase(text)

def jaccard_similarity(str1, str2):
    set1 = set(str1.lower().split())
    set2 = set(str2.lower().split())
    return len(set1.intersection(set2)) / len(set1.union(set2))

def highlight_matching_words(sentence, input_words):
    words = sentence.split()
    highlighted_sentence = " ".join([f"<span style='background-color: yellow;'>{word}</span>" if word.lower() in input_words else word for word in words])
    return highlighted_sentence

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server


app.layout = html.Div([
    # Header
    html.Div([
        html.Img(src='/assets/1.png', id='logo', style={'height': '80px'}),
        html.H2('BIT Textual Similarity Dashboard: A Jaccard Index Approach', style={'marginLeft': '20px'})
    ], style={'display': 'flex', 'alignItems': 'center', 'background': '#f4f4f4', 'borderBottom': 'solid 1px #ddd', 'padding': '10px'}),
    
    # Main Content
    html.Div([
        # Dashboard 1
        html.Div([
            # Inputs for Dashboard 1
            html.Div([
                html.Label('Enter Sentence:'),
                dcc.Textarea(id='input-sentence', value='', style={'width': '100%', 'height': '100px', 'marginBottom': '20px'}),
                html.Button('Search', id='search-button', n_clicks=0),
                html.Br(),  # New line for "Top N Similarities"
                html.Label('Top N Similarities:'),
                dcc.Input(id='input-top-n', value='5', type='number', style={'width': '100px', 'marginBottom': '20px'}),
                html.Label('Threshold (between 0 and 1):', style={'marginRight': '10px'}),  # Add space
                dcc.Input(id='input-threshold', value='0.5', type='number', min=0, max=1, step=0.05, style={'marginBottom': '40px'})
            ], style={'width': '30%', 'padding': '20px', 'borderRight': 'solid 1px #ddd'}),
            
            # Outputs for Dashboard 1
            html.Div([
                html.Label('Similarity Table:'),
                dcc.Loading(
                    id="loading",
                    type="default",
                    children=[
                        dash_table.DataTable(id='similarity-table'),
                        dcc.Graph(id='similarity-graph', style={'marginTop': '20px'})
                    ]
                )
            ], style={'width': '70%', 'padding': '20px'})
        ], style={'display': 'flex', 'flexDirection': 'row', 'borderBottom': 'solid 1px #ddd', 'padding': '10px'}),
        
        # Dashboard 2
        html.Div([
            # Outputs for Dashboard 2
            html.Div([
                html.Label('Most Similar Treaty:'),
                html.Div(id='similarity-output', style={'marginTop': '20px'})
            ], style={'padding': '20px'})
        ]),
    ], style={'maxWidth': '1200px', 'margin': '0 auto'}),
    
    # Footer
    html.Div([
        html.P("© 2023 The University of Sydney, Email: haohui.lu@sydney.edu.au", style={'textAlign': 'center', 'margin': '20px', 'fontSize': '12px', 'color': '#999'})
    ], style={'borderTop': '1px solid #ddd', 'background': '#f4f4f4'}),
])


# Callbacks that use common input
@app.callback(
    [Output('similarity-graph', 'figure'),
     Output('similarity-table', 'data'),
     Output('similarity-output', 'children')],
    Input('search-button', 'n_clicks'),
    Input('input-sentence', 'value'),
    Input('input-top-n', 'value'),
    Input('input-threshold', 'value')
)
def update_outputs(n_clicks, input_sentence, top_n, threshold):
    # For Dashboard 1
    if n_clicks == 0 or not input_sentence:
        return px.scatter(), [], None

    df = pd.read_csv("BIT_all.csv")
    df['pp_text'] = df['treaty_article_sentence'].apply(lambda x: preprocessing(x) if isinstance(x, str) else np.NaN)
    similarities = df['pp_text'].apply(lambda x: jaccard_similarity(x, preprocessing(input_sentence)))
    df['similarity'] = round(similarities,4)

    most_similar_row = df.sort_values(by='similarity', ascending=False).iloc[0]
    input_words = set(preprocessing(input_sentence).split())
    highlighted_sentence = highlight_matching_words(most_similar_row['treaty_article_sentence'], input_words)
    similarity_output = html.Div([
        html.H5(f"Name: {most_similar_row['Name']}"),
        html.P("Similar Sentence: ", style={'font-weight': 'bold'}),
        DangerouslySetInnerHTML(highlighted_sentence)
    ])

    # Create similarity table and graph as previously
    threshold_filtered = df[df['similarity'] >= float(threshold)]
    top_n_similarity = threshold_filtered.sort_values(by='similarity', ascending=False).head(int(top_n))
    fig = px.scatter(top_n_similarity, x='Date of signature', y='similarity')
    annotations = []
    for index, row in top_n_similarity.iterrows():
        annotations.append(
            dict(
                x=row['Date of signature'],
                y=row['similarity'],
                xref='x',
                yref='y',
                text=row['Name'],
                showarrow=False,
                font=dict(
                    size=10
                ),
                ax=0,
                ay=-40,
                textangle=-45
            )
        )
    fig.update_layout(annotations=annotations)

    table_data = top_n_similarity[["ID", "Name", "location", 'similarity']]
    
    return fig, table_data.to_dict('records'), similarity_output

if __name__ == '__main__':
    app.run_server()
