import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Heatmap
import joblib
import sqlite3


app = Flask(__name__)


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


# load data

conn = sqlite3.connect('../data/DisasterResponse.db')
df = pd.read_sql("SELECT * FROM DisasterResponse", con=conn)
conn.close()


# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    df_categories = df.drop(df.iloc[:, 0:4], axis=1)

    category_counts = pd.DataFrame(df_categories.sum().sort_values(), columns=['counts'])

    df_categories_corr = df_categories.corr().dropna(axis=0, how='all').dropna(axis=1, how='all')

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        # chart 1: genre bar chart
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        # chart 2: categories bar chart
        {
            'data': [
                Bar(
                    x=category_counts.index,
                    y=category_counts['counts']
                )
            ],

            'layout': {
                'title': 'Distribution of Categories per Classification',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
        # chart 3: categories heatmap
        {
            'data': [
                Heatmap(
                    z = df_categories_corr,
                    x =df_categories_corr.columns,
                    y = df_categories_corr.index
                )
            ],

            'layout': {
                'title': 'Correlation of Category Counts',
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()
