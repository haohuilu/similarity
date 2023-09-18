# Imports
import dash
from dash import dcc, html, callback_context
from dash.dependencies import Input, Output, State
from dash import dash_table
from dash_table.Format import Format
import dash_bootstrap_components as dbc

import pandas as pd
import plotly.express as px
import re
import string
import numpy as np
import nltk



# Pre-processing functions
def remove_urls(text):
    new_text = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", text).split())
    return new_text

def text_lowercase(text):
    return text.lower()

def remove_numbers(text):
    result = re.sub(r'\d+', '', text)
    return result

def remove_punctuation(text):
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)

def preprocessing(text):
    text = text_lowercase(text)
    text = remove_urls(text)
    text = remove_numbers(text)
    text = remove_punctuation(text)
    return text

def jaccard_similarity(str1, str2):
    set1 = set(str1.lower().split())
    set2 = set(str2.lower().split())
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

app.layout = html.Div([
    # Outermost Container
    html.Div([
        # Header with Logo and Title
        html.Div([
            html.Img(src='/assets/1.png', id='logo', style={'height': '80px'}),
            html.H2('BIT Textual Similarity Dashboard: A Jaccard Index Approach', style={'marginLeft': '20px'})
        ], style={'display': 'flex', 'alignItems': 'center', 'background': '#f4f4f4', 'borderBottom': 'solid 1px #ddd', 'padding': '10px'}),
        
        # Main Content Container
        html.Div([
            # Left Side
            html.Div([
                html.Div(style={'height': '50px'}),  # Spacer
                html.Label('Enter New Sentence:'),
                dcc.Textarea(id='input-sentence', value='', style={'width': '100%', 'height': '100px', 'marginBottom': '20px'}),
                
                html.Label('Top N Similarities:'),
                dcc.Input(id='input-top-n', value='5', type='number', style={'marginBottom': '20px'}),
                
                html.Label('Threshold (between 0 and 1):'),
                dcc.Input(id='input-threshold', value='0.5', type='number', min=0, max=1, step=0.05, style={'marginBottom': '40px'}),
                
                html.Div(style={'height': '20px'}),  # <-- This is the spacer
                html.Button('Search', id='search-button', n_clicks=0)
            ], style={'width': '25%', 'padding': '40px', 'borderRight': 'solid 1px #ddd'}),
            
            # Right Side
            html.Div([
                html.Label('Similarity Table:'),
                dcc.Loading(  # <-- Added this loading spinner component
                    id="loading",
                    type="default",
                    children=[
                        dash_table.DataTable(id='similarity-table'),
                        dcc.Graph(id='similarity-graph', style={'marginTop': '20px'})
                    ]
                )
            ], style={'width': '75%', 'padding': '20px'})
        ], style={'display': 'flex', 'flexDirection': 'row'}),
    ], style={'maxWidth': '1200px', 'margin': '0 auto'}),  # Centralize the content
    
    # Footer
    html.Div([
        html.P("© 2023 The University of Sydney, Email: haohui.lu@sydney.edu.au", style={'textAlign': 'center', 'margin': '20px', 'fontSize': '12px', 'color': '#999'})
    ], style={'borderTop': '1px solid #ddd', 'background': '#f4f4f4'}),
])

@app.callback(
    [Output('similarity-graph', 'figure'),
     Output('similarity-table', 'data')],
    Input('search-button', 'n_clicks'),
    Input('input-sentence', 'value'),
    Input('input-top-n', 'value'),
    Input('input-threshold', 'value')
)
def update_graph(n_clicks, input_sentence, top_n, threshold):
    if n_clicks == 0 or not input_sentence:
        return px.scatter(), []

    df = pd.read_csv("BIT.csv")
    pp_text = [preprocessing(text) if isinstance(text, str) else np.NaN for text in df['treaty_article_sentence']]
    df['pp_text'] = pp_text

    new_sentence_pp = preprocessing(input_sentence)
    new_row = pd.DataFrame({'pp_text': [new_sentence_pp]})
    df = pd.concat([df, new_row], ignore_index=True)

    results = [
        {
            'BIT1': i, 
            'BIT2': j, 
            'similarity': round(jaccard_similarity(df['pp_text'][i], df['pp_text'][j]), 4)  # <-- Rounded to 4 decimal places
        }
        for i in df.index for j in range(i+1, len(df))
    ]
    sim_df = pd.DataFrame(results)
    last_bit = max(sim_df['BIT2'].max(), sim_df['BIT1'].max())
    last_bit_df = sim_df[(sim_df['BIT1'] == last_bit) | (sim_df['BIT2'] == last_bit)]
    threshold_filtered = last_bit_df[last_bit_df['similarity'] >= float(threshold)]
    top_n_similarity = threshold_filtered.sort_values(by='similarity', ascending=False).head(int(top_n))
    merged_df = top_n_similarity.merge(df, left_on='BIT1', right_index=True)
    merged_df['Date of signature'] = pd.to_datetime(merged_df['Date of signature'], dayfirst=True)
    fig = px.scatter(merged_df, x='Date of signature', y='similarity', text='Name')
    fig.update_traces(textposition='top center')

    fig.update_layout(title='Timeline of BITs with Similarity')
    
    similarity_df = merged_df[["Name" ,'similarity']]
    
    return fig, similarity_df.to_dict('records')

if __name__ == '__main__':
    app.run_server()