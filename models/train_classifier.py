import sys
import pandas as pd
import sqlite3
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV


nltk.download(['punkt', 'wordnet'])


def load_data(database_filepath):
    """
    :param database_filepath:
    :return: target variable, features, feature names
    """
    conn = sqlite3.connect(database_filepath)
    df = pd.read_sql('SELECT * FROM DisasterResponse', con=conn)
    conn.close()

    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    category_names = Y.columns.values

    return X, Y, category_names


def tokenize(text):
    """
    :param text:
    :return: tokenized and cleaned text
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    # nlp pipeline
    text_processing = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer())
    ])

    ab = AdaBoostClassifier()

    pipeline = Pipeline([('pre_pipeline', text_processing),
                         ('clf', MultiOutputClassifier(ab))])

    parameters = {
        'clf__estimator__learning_rate': [0.5, 1.0],
        'clf__estimator__n_estimators': [10, 20]
    }

    model = GridSearchCV(pipeline, param_grid=parameters, cv=5, n_jobs=-1, verbose=3)

    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """
    :param model:
    :param X_test:
    :param Y_test:
    :param category_names:
    :return: prints the classification_report for the model
    """
    y_prediction_test = model.predict(X_test)

    return print(classification_report(Y_test.values, y_prediction_test, target_names=category_names))


def save_model(model, model_filepath):
    """
    :param model:
    :param model_filepath:
    :return: save model in .pkl file
    """
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
